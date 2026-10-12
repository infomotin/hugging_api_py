from datetime import datetime
import secrets
import hashlib
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from core.database import Base

class ApiKey(Base):
    __tablename__ = 'api_keys'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    name = Column(String(100), nullable=False)
    key_prefix = Column(String(16), nullable=False, index=True)  # e.g. "sk-ent-a1b2..."
    key_hash = Column(String(64), nullable=False, index=True)    # SHA-256 hash of raw key
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)

    user = relationship('User', backref='api_keys', lazy='joined')

    @staticmethod
    def generate_key_pair(name: str, user_id: int):
        """Generates a raw secret key and creates an ApiKey instance."""
        raw_token = f"sk-ent-{secrets.token_hex(16)}"
        key_prefix = raw_token[:12] + "..."
        key_hash = hashlib.sha256(raw_token.encode()).hexdigest()

        key_instance = ApiKey(
            user_id=user_id,
            name=name,
            key_prefix=key_prefix,
            key_hash=key_hash,
            is_active=True
        )
        return raw_token, key_instance

    @staticmethod
    def hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.strip().encode()).hexdigest()

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "key_prefix": self.key_prefix,
            "is_active": self.is_active,
            "created_at": self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            "last_used_at": self.last_used_at.strftime('%Y-%m-%d %H:%M:%S') if self.last_used_at else "Never"
        }
