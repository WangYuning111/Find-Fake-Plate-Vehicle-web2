#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
套牌车智能稽查系统 - 生产级入口

改进点：
  - 配置管理集中化
  - SQLite 数据库持久化历史记录
  - 车辆信息支持 CRUD
  - 结果图片可视化（检测框+标签）
  - 完善的错误处理和状态码
  - 支持批量上传
  - 请求限流与文件大小限制
"""
import os
import time
import uuid
from datetime import datetime

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from config import Config
from database import (
    add_record, get_records, get_record_count, get_stats,
    add_vehicle, update_vehicle, delete_vehicle, list_vehicles,
    import_csv_to_db
)
from inference import load_models, predict_single
from feedback_db import save_feedback, list_pending_feedback, update_feedback, get_feedback_stats
from improve_accuracy import enhance_image_for_plate

# ============== 初始化 ==============
Config.ensure_dirs()
app = Flask(__name__)
app.config.from_object(Config)
app.send_file_max_age_default = Config.MAX_CONTENT_LENGTH

# 启动时预热模型（可选）
# load_models()


# ============== 工具函数 ==============
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def save_upload_file(file_obj):
    """保存上传文件，返回唯一文件名和路径"""
    ext = file_obj.filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex[:16]}_{int(time.time())}.{ext}"
    filepath = os.path.join(Config.UPLOAD_FOLDER, unique_name)
    file_obj.save(filepath)
    return unique_name, filepath


# ============== 页面路由 ==============
@app.route('/', methods=['GET', 'POST'])
def index():
    """首页：GET 显示页面，POST 处理上传并显示结果"""
    stats = get_stats()

    if request.method == 'GET':
        return render_template('index.html', stats=stats)

    # POST 处理
    if 'file' not in request.files:
        return render_template('index.html', error='请选择要上传的图片', stats=stats)

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('index.html', error='文件无效或不支持的格式', stats=stats)

    try:
        unique_name, upload_path = save_upload_file(file)
        result_name = f"result_{unique_name}"
        result_path = os.path.join(Config.RESULT_FOLDER, result_name)

        result = predict_single(upload_path, save_result_path=result_path)

        if result['error']:
            return render_template('index.html', error=result['error'], stats=stats)

        add_record(
            image_path=upload_path,
            plate=result['plate_number'],
            v_type=result['vehicle_type'],
            v_color=result['vehicle_color'],
            brand=result['vehicle_brand'],
            is_fake=result['is_fake'],
            true_brand=result['true_brand'],
            confidence=result['confidence'],
            result_image=result_path
        )

        # 保存反馈数据（用于后续精度提升）
        try:
            save_feedback(upload_path, result['plate_number'], result['vehicle_type'],
                          result['vehicle_color'], result['vehicle_brand'], result['is_fake'])
        except Exception as e:
            app.logger.warning(f"保存反馈数据失败: {e}")

        # 构造 result_image 的 URL
        result_image_url = url_for('static', filename=f'results/{result_name}')

        return render_template('index.html',
                               result=result,
                               result_image=result_image_url,
                               stats=get_stats())
    except Exception as e:
        app.logger.error(f"分析失败: {e}", exc_info=True)
        return render_template('index.html', error=f'分析失败: {str(e)}', stats=stats)


@app.route('/feedback')
def feedback_page():
    """反馈数据修正页面"""
    records = list_pending_feedback(limit=50)
    stats = get_feedback_stats()
    return render_template('feedback.html', records=records, stats=stats)


@app.route('/api/feedback/<fid>/correct', methods=['POST'])
def api_correct_feedback(fid):
    """修正单条反馈数据"""
    data = request.get_json() or {}
    ok = update_feedback(
        fid,
        corrected_plate=data.get('plate', ''),
        corrected_brand=data.get('brand', ''),
        corrected_type=data.get('type', ''),
        corrected_color=data.get('color', '')
    )
    return jsonify({'success': ok})


@app.route('/history')
def history():
    """识别历史记录页面"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    offset = (page - 1) * per_page
    records = get_records(limit=per_page, offset=offset)
    total = get_record_count()
    total_pages = (total + per_page - 1) // per_page
    return render_template('history.html',
                           records=records,
                           page=page,
                           total_pages=total_pages,
                           total=total)


@app.route('/database')
def vehicle_db_page():
    """车辆数据库管理页面"""
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '').strip()
    per_page = 20
    offset = (page - 1) * per_page
    vehicles, total = list_vehicles(search=search or None, limit=per_page, offset=offset)
    total_pages = (total + per_page - 1) // per_page
    return render_template('database.html',
                           vehicles=vehicles,
                           page=page,
                           total_pages=total_pages,
                           total=total,
                           search=search)


# ============== API 路由 ==============
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """单张图片分析 API"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '文件名为空'}), 400
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': '不支持的文件格式'}), 400

    try:
        # 保存上传文件
        unique_name, upload_path = save_upload_file(file)

        # 结果图路径
        result_name = f"result_{unique_name}"
        result_path = os.path.join(Config.RESULT_FOLDER, result_name)

        # 推理
        result = predict_single(upload_path, save_result_path=result_path)

        if result['error']:
            return jsonify({'success': False, 'error': result['error']}), 422

        # 写入数据库
        add_record(
            image_path=upload_path,
            plate=result['plate_number'],
            v_type=result['vehicle_type'],
            v_color=result['vehicle_color'],
            brand=result['vehicle_brand'],
            is_fake=result['is_fake'],
            true_brand=result['true_brand'],
            confidence=result['confidence'],
            result_image=result_path
        )

        # 保存反馈数据
        try:
            save_feedback(upload_path, result['plate_number'], result['vehicle_type'],
                          result['vehicle_color'], result['vehicle_brand'], result['is_fake'])
        except Exception:
            pass

        return jsonify({
            'success': True,
            'data': {
                'plate_number': result['plate_number'],
                'vehicle_type': result['vehicle_type'],
                'vehicle_color': result['vehicle_color'],
                'vehicle_brand': result['vehicle_brand'],
                'is_fake': result['is_fake'],
                'true_brand': result['true_brand'],
                'confidence': result['confidence'],
                'process_time': result['process_time'],
                'result_image_url': f'/static/results/{result_name}' if result['result_image'] else None,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器内部错误: {str(e)}'}), 500


@app.route('/api/analyze', methods=['GET'])
def api_analyze_get():
    """兼容 GET 请求的表单上传（供前端 form 使用）"""
    if 'file' not in request.files:
        return render_template('index.html', error='请选择要上传的图片')

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('index.html', error='文件无效或不支持的格式')

    try:
        unique_name, upload_path = save_upload_file(file)
        result_name = f"result_{unique_name}"
        result_path = os.path.join(Config.RESULT_FOLDER, result_name)

        result = predict_single(upload_path, save_result_path=result_path)

        if result['error']:
            return render_template('index.html', error=result['error'], stats=get_stats())

        add_record(
            image_path=upload_path,
            plate=result['plate_number'],
            v_type=result['vehicle_type'],
            v_color=result['vehicle_color'],
            brand=result['vehicle_brand'],
            is_fake=result['is_fake'],
            true_brand=result['true_brand'],
            confidence=result['confidence'],
            result_image=result_path
        )

        # 保存反馈数据
        try:
            save_feedback(upload_path, result['plate_number'], result['vehicle_type'],
                          result['vehicle_color'], result['vehicle_brand'], result['is_fake'])
        except Exception:
            pass

        return render_template('index.html',
                               result=result,
                               result_image=f'/static/results/{result_name}',
                               stats=get_stats())

    except Exception as e:
        return render_template('index.html', error=f'处理失败: {str(e)}', stats=get_stats())


@app.route('/api/batch_analyze', methods=['POST'])
def api_batch_analyze():
    """批量分析 API"""
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'success': False, 'error': '没有上传文件'}), 400

    results = []
    for file in files:
        if not allowed_file(file.filename):
            results.append({'filename': file.filename, 'error': '不支持的格式'})
            continue

        try:
            unique_name, upload_path = save_upload_file(file)
            result_name = f"result_{unique_name}"
            result_path = os.path.join(Config.RESULT_FOLDER, result_name)

            res = predict_single(upload_path, save_result_path=result_path)

            if not res['error']:
                add_record(
                    image_path=upload_path,
                    plate=res['plate_number'],
                    v_type=res['vehicle_type'],
                    v_color=res['vehicle_color'],
                    brand=res['vehicle_brand'],
                    is_fake=res['is_fake'],
                    true_brand=res['true_brand'],
                    confidence=res['confidence'],
                    result_image=result_path
                )

            results.append({
                'filename': file.filename,
                'success': res['error'] is None,
                'error': res['error'],
                'data': {
                    'plate_number': res['plate_number'],
                    'vehicle_type': res['vehicle_type'],
                    'vehicle_color': res['vehicle_color'],
                    'vehicle_brand': res['vehicle_brand'],
                    'is_fake': res['is_fake'],
                    'result_image_url': f'/static/results/{result_name}' if res['result_image'] else None,
                }
            })
        except Exception as e:
            results.append({'filename': file.filename, 'success': False, 'error': str(e)})

    return jsonify({'success': True, 'results': results})


# ============== 车辆数据库 API ==============
@app.route('/api/vehicles', methods=['GET'])
def api_list_vehicles():
    """查询车辆列表"""
    search = request.args.get('search', '').strip()
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    vehicles, total = list_vehicles(search=search or None, limit=per_page, offset=(page - 1) * per_page)
    return jsonify({'success': True, 'data': vehicles, 'total': total, 'page': page})


@app.route('/api/vehicles', methods=['POST'])
def api_add_vehicle():
    """添加车辆"""
    data = request.get_json() or {}
    plate_no = data.get('plate_no', '').strip()
    car_brand = data.get('car_brand', '').strip()

    if not plate_no or not car_brand:
        return jsonify({'success': False, 'error': '车牌号和品牌不能为空'}), 400

    success = add_vehicle(
        plate_no=plate_no,
        car_brand=car_brand,
        car_type=data.get('car_type'),
        owner_name=data.get('owner_name'),
        registration_date=data.get('registration_date'),
        status=data.get('status', '正常')
    )

    if success:
        return jsonify({'success': True, 'message': '添加成功'})
    return jsonify({'success': False, 'error': '车牌号已存在'}), 409


@app.route('/api/vehicles/<plate_no>', methods=['PUT'])
def api_update_vehicle(plate_no):
    """更新车辆信息"""
    data = request.get_json() or {}
    success = update_vehicle(plate_no, **data)
    if success:
        return jsonify({'success': True, 'message': '更新成功'})
    return jsonify({'success': False, 'error': '车辆不存在'}), 404


@app.route('/api/vehicles/<plate_no>', methods=['DELETE'])
def api_delete_vehicle(plate_no):
    """删除车辆信息"""
    success = delete_vehicle(plate_no)
    if success:
        return jsonify({'success': True, 'message': '删除成功'})
    return jsonify({'success': False, 'error': '车辆不存在'}), 404


@app.route('/api/vehicles/import_csv', methods=['POST'])
def api_import_csv():
    """重新导入 CSV"""
    try:
        import_csv_to_db()
        return jsonify({'success': True, 'message': '导入成功'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============== 通用 API ==============
@app.route('/api/stats', methods=['GET'])
def api_stats():
    """获取统计信息"""
    return jsonify({'success': True, 'data': get_stats()})


@app.route('/api/records', methods=['GET'])
def api_records():
    """分页获取识别记录"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    records = get_records(limit=per_page, offset=(page - 1) * per_page)
    return jsonify({'success': True, 'data': records, 'total': get_record_count()})


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'service': '套牌车智能稽查系统',
        'version': '2.0',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })


@app.route('/warmup', methods=['GET'])
def warmup():
    """预热模型"""
    try:
        load_models()
        return jsonify({'success': True, 'message': '模型预热完成'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============== 静态文件 ==============
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(Config.UPLOAD_FOLDER, filename)


# ============== 错误处理 ==============
@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': '文件过大，最大支持 16MB'}), 413


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': '接口不存在'}), 404
    return render_template('index.html', error='页面不存在'), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': '服务器内部错误'}), 500


# ============== 启动 ==============
if __name__ == '__main__':
    print("=" * 60)
    print("  套牌车智能稽查系统 V2.0")
    print("=" * 60)
    print(f"  访问地址: http://{Config.HOST}:{Config.PORT}")
    print(f"  调试模式: {Config.DEBUG}")
    print(f"  数据目录: {os.path.dirname(Config.DATABASE_PATH)}")
    print("=" * 60)
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG, threaded=True)
