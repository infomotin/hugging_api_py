from datetime import datetime
from flask import jsonify, request, current_app
from flask_login import current_user, login_user
from app.api import api_bp
from app.models import db, User, Role, Permission, Resource, AuditLog
from app.auth.decorators import role_required, permission_required, admin_required

@api_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint exposing system info and DB status."""
    db_status = "connected"
    try:
        db.session.execute(db.text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {str(e)}"

    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'database': db_status,
        'app_name': 'Flask RBAC Enterprise API',
        'version': '1.0.0'
    }), 200


@api_bp.route('/auth/login', methods=['POST'])
def api_login():
    """API endpoint for authentication."""
    data = request.get_json() or {}
    identifier = data.get('username') or data.get('email')
    password = data.get('password')

    if not identifier or not password:
        return jsonify({'error': 'Bad Request', 'message': 'Username/email and password are required'}), 400

    user = User.query.filter(
        (User.username == identifier) | (User.email == identifier)
    ).first()

    if not user or not user.check_password(password):
        return jsonify({'error': 'Unauthorized', 'message': 'Invalid credentials'}), 401

    if not user.is_active:
        return jsonify({'error': 'Forbidden', 'message': 'Account is deactivated'}), 403

    user.last_login = datetime.utcnow()
    login_user(user)

    # Log audit
    log = AuditLog(
        user_id=user.id,
        username=user.username,
        action='API_LOGIN',
        details=f'User {user.username} authenticated via API endpoint.',
        ip_address=request.remote_addr
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({
        'message': 'Authentication successful',
        'user': user.to_dict(),
        'permissions': [p.name for p in user.role.permissions] if user.role else []
    }), 200


@api_bp.route('/auth/me', methods=['GET'])
def api_me():
    """Return currently authenticated user profile and permissions."""
    if not current_user.is_authenticated:
        return jsonify({'error': 'Unauthorized', 'message': 'No active session'}), 401

    return jsonify({
        'user': current_user.to_dict(),
        'role': current_user.role.name if current_user.role else None,
        'permissions': [p.name for p in current_user.role.permissions] if current_user.role else []
    }), 200


@api_bp.route('/stats', methods=['GET'])
@permission_required('view_dashboard')
def api_stats():
    """High-level metrics for dashboard telemetry."""
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    total_resources = Resource.query.count()
    roles_breakdown = {}
    for r in Role.query.all():
        roles_breakdown[r.name] = len(r.users)

    recent_logs = [log.to_dict() for log in AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(5).all()]

    return jsonify({
        'metrics': {
            'total_users': total_users,
            'active_users': active_users,
            'total_resources': total_resources,
            'roles_count': Role.query.count(),
        },
        'roles_breakdown': roles_breakdown,
        'recent_activity': recent_logs
    }), 200


@api_bp.route('/users', methods=['GET'])
@permission_required('manage_users')
def api_get_users():
    """List all registered users (Requires manage_users)."""
    users = User.query.order_by(User.id.asc()).all()
    return jsonify({
        'count': len(users),
        'users': [u.to_dict() for u in users]
    }), 200


@api_bp.route('/users', methods=['POST'])
@admin_required
def api_create_user():
    """Create a new user with specific role (Admin only)."""
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password')
    role_name = data.get('role', 'User')

    if not username or not email or not password:
        return jsonify({'error': 'Bad Request', 'message': 'username, email, and password required'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Conflict', 'message': 'Username already exists'}), 409

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Conflict', 'message': 'Email already registered'}), 409

    role = Role.query.filter_by(name=role_name).first()
    if not role:
        return jsonify({'error': 'Bad Request', 'message': f'Role {role_name} not found'}), 400

    new_user = User(
        username=username,
        email=email,
        role=role,
        is_active=data.get('is_active', True)
    )
    new_user.set_password(password)

    db.session.add(new_user)
    db.session.flush()

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action='API_CREATE_USER',
        details=f'Admin created user {username} ({role_name})',
        ip_address=request.remote_addr
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({
        'message': f'User {username} created successfully',
        'user': new_user.to_dict()
    }), 201


@api_bp.route('/users/<int:user_id>/role', methods=['PUT'])
@admin_required
def api_update_user_role(user_id):
    """Change a user's role (Admin only)."""
    user = User.query.get_or_404(user_id)
    data = request.get_json() or {}
    role_name = data.get('role')

    if not role_name:
        return jsonify({'error': 'Bad Request', 'message': 'role parameter required'}), 400

    role = Role.query.filter_by(name=role_name).first()
    if not role:
        return jsonify({'error': 'Bad Request', 'message': f'Role {role_name} not found'}), 400

    old_role = user.role.name if user.role else 'None'
    user.role = role

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action='ROLE_UPDATED',
        details=f'Changed {user.username} role from {old_role} to {role_name}',
        ip_address=request.remote_addr
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({
        'message': f'User role updated to {role_name}',
        'user': user.to_dict()
    }), 200


@api_bp.route('/users/<int:user_id>/status', methods=['PUT'])
@permission_required('manage_users')
def api_toggle_user_status(user_id):
    """Enable or disable a user account."""
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        return jsonify({'error': 'Forbidden', 'message': 'You cannot deactivate your own account'}), 400

    data = request.get_json() or {}
    new_status = data.get('is_active', not user.is_active)
    user.is_active = new_status

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action='STATUS_UPDATED',
        details=f'Toggled {user.username} active status to {new_status}',
        ip_address=request.remote_addr
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({
        'message': f'User active status set to {new_status}',
        'user': user.to_dict()
    }), 200


@api_bp.route('/users/<int:user_id>', methods=['DELETE'])
@permission_required('delete_users')
def api_delete_user(user_id):
    """Delete a user account."""
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        return jsonify({'error': 'Forbidden', 'message': 'You cannot delete yourself'}), 400

    username = user.username
    db.session.delete(user)

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action='USER_DELETED',
        details=f'Permanently deleted user: {username}',
        ip_address=request.remote_addr
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({'message': f'User {username} deleted successfully'}), 200


@api_bp.route('/roles', methods=['GET'])
@permission_required('manage_users')
def api_get_roles():
    """List roles with assigned permissions."""
    roles = Role.query.all()
    return jsonify({
        'roles': [r.to_dict() for r in roles]
    }), 200


@api_bp.route('/permissions', methods=['GET'])
@permission_required('manage_roles')
def api_get_permissions():
    """List all registered system permissions."""
    perms = Permission.query.all()
    return jsonify({
        'permissions': [p.to_dict() for p in perms]
    }), 200


@api_bp.route('/resources', methods=['GET'])
def api_get_resources():
    """List resources. Accessible to all users."""
    resources = Resource.query.order_by(Resource.created_at.desc()).all()
    return jsonify({
        'count': len(resources),
        'resources': [r.to_dict() for r in resources]
    }), 200


@api_bp.route('/resources', methods=['POST'])
@permission_required('create_resources')
def api_create_resource():
    """Create a new resource record."""
    data = request.get_json() or {}
    title = data.get('title', '').strip()
    description = data.get('description', '')
    category = data.get('category', 'General')
    status = data.get('status', 'Published')

    if not title:
        return jsonify({'error': 'Bad Request', 'message': 'Title is required'}), 400

    res = Resource(
        title=title,
        description=description,
        category=category,
        status=status,
        created_by=current_user.id
    )
    db.session.add(res)
    db.session.flush()

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action='RESOURCE_CREATED',
        details=f'Created resource: {title}',
        ip_address=request.remote_addr
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({
        'message': 'Resource created successfully',
        'resource': res.to_dict()
    }), 201


@api_bp.route('/resources/<int:res_id>', methods=['DELETE'])
@permission_required('delete_resources')
def api_delete_resource(res_id):
    """Delete a resource."""
    res = Resource.query.get_or_404(res_id)
    title = res.title
    db.session.delete(res)

    log = AuditLog(
        user_id=current_user.id,
        username=current_user.username,
        action='RESOURCE_DELETED',
        details=f'Deleted resource: {title}',
        ip_address=request.remote_addr
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({'message': f'Resource "{title}" deleted successfully'}), 200


@api_bp.route('/audit-logs', methods=['GET'])
@permission_required('view_audit_logs')
def api_get_audit_logs():
    """View system audit events."""
    limit = min(int(request.args.get('limit', 50)), 100)
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
    return jsonify({
        'count': len(logs),
        'logs': [l.to_dict() for l in logs]
    }), 200
