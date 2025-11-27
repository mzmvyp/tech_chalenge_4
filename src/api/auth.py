"""
Módulo de Autenticação JWT
===========================

Sistema de autenticação usando JWT (JSON Web Tokens) para proteger a API.

Fluxo:
1. Cliente envia API Key para /auth/login
2. API valida e retorna JWT (access token)
3. Cliente usa JWT no header Authorization: Bearer <token>
4. API valida JWT em cada requisição protegida

Autor: Tech Challenge - Fase 04
Data: 2024-11-27
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# ============================================
# CONFIGURAÇÕES
# ============================================

# Chave secreta para assinar JWT (deve ser segura e única)
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production-use-secure-random-key")

# Algoritmo de criptografia
ALGORITHM = "HS256"

# Tempo de expiração do token (padrão: 24 horas)
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

# API Keys válidas (em produção, usar banco de dados)
# Formato: {api_key: {"name": "nome", "active": True}}
API_KEYS = {
    os.getenv("API_KEY_1", "default-api-key-change-me"): {
        "name": "Default API Key",
        "active": True,
        "permissions": ["read", "predict"]
    }
}

# Contexto para hash de senhas (se necessário no futuro)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme para FastAPI
security = HTTPBearer()


# ============================================
# FUNÇÕES DE JWT
# ============================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Cria um JWT access token.
    
    Args:
        data: Dados para incluir no token (ex: {"sub": "api_key"})
        expires_delta: Tempo de expiração (opcional)
    
    Returns:
        Token JWT codificado
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """
    Verifica e decodifica um JWT token.
    
    Args:
        token: Token JWT para verificar
    
    Returns:
        Payload decodificado do token
    
    Raises:
        HTTPException: Se o token for inválido ou expirado
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido ou expirado",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================
# VALIDAÇÃO DE API KEY
# ============================================

def verify_api_key(api_key: str) -> dict:
    """
    Verifica se uma API key é válida.
    
    Args:
        api_key: API key para verificar
    
    Returns:
        Informações da API key se válida
    
    Raises:
        HTTPException: Se a API key for inválida
    """
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida"
        )
    
    api_key_info = API_KEYS[api_key]
    
    if not api_key_info.get("active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key desativada"
        )
    
    return api_key_info


# ============================================
# DEPENDENCY PARA AUTENTICAÇÃO
# ============================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Dependency para verificar autenticação em endpoints protegidos.
    
    Uso:
        @app.get("/protected")
        async def protected_route(current_user: dict = Depends(get_current_user)):
            ...
    
    Args:
        credentials: Credenciais do header Authorization
    
    Returns:
        Payload do token (informações do usuário)
    
    Raises:
        HTTPException: Se não autenticado
    """
    token = credentials.credentials
    payload = verify_token(token)
    return payload


# ============================================
# FUNÇÕES AUXILIARES
# ============================================

def generate_secret_key() -> str:
    """
    Gera uma chave secreta segura para JWT.
    
    Returns:
        Chave secreta aleatória (64 caracteres hex)
    """
    import secrets
    return secrets.token_urlsafe(64)


def add_api_key(api_key: str, name: str, permissions: list = None) -> None:
    """
    Adiciona uma nova API key ao sistema.
    
    Args:
        api_key: Chave da API
        name: Nome/descrição da chave
        permissions: Lista de permissões (ex: ["read", "predict"])
    """
    if permissions is None:
        permissions = ["read", "predict"]
    
    API_KEYS[api_key] = {
        "name": name,
        "active": True,
        "permissions": permissions
    }


def list_api_keys() -> dict:
    """
    Lista todas as API keys (sem mostrar as chaves completas por segurança).
    
    Returns:
        Dicionário com informações das API keys
    """
    return {
        key[:8] + "..." + key[-4:]: info
        for key, info in API_KEYS.items()
    }

