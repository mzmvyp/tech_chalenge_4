"""
Script para Gerar Chaves Secretas
==================================

Gera chaves secretas seguras para JWT e API Keys.

Uso:
    python scripts/generate_secrets.py

Autor: Tech Challenge - Fase 04
Data: 2024-11-27
"""

import secrets
import sys
from pathlib import Path

def generate_jwt_secret_key() -> str:
    """Gera uma chave secreta segura para JWT (64 bytes, URL-safe)."""
    return secrets.token_urlsafe(64)


def generate_api_key() -> str:
    """Gera uma API key segura (32 bytes, URL-safe)."""
    return secrets.token_urlsafe(32)


def main():
    """Gera e exibe chaves secretas."""
    print("\n" + "="*60)
    print("🔐 GERADOR DE CHAVES SECRETAS")
    print("="*60)
    print()
    
    # Gerar chaves
    jwt_secret = generate_jwt_secret_key()
    api_key = generate_api_key()
    
    print("✅ Chaves geradas com sucesso!")
    print()
    print("="*60)
    print("📋 ADICIONE AO SEU ARQUIVO .env:")
    print("="*60)
    print()
    print(f"JWT_SECRET_KEY={jwt_secret}")
    print(f"API_KEY_1={api_key}")
    print()
    print("="*60)
    print("⚠️  IMPORTANTE:")
    print("="*60)
    print("1. NUNCA compartilhe essas chaves")
    print("2. NUNCA faça commit do arquivo .env no Git")
    print("3. Use chaves diferentes para desenvolvimento e produção")
    print("4. Guarde essas chaves em local seguro")
    print()
    
    # Perguntar se quer salvar em .env
    save = input("Deseja salvar automaticamente no arquivo .env? (s/n): ").strip().lower()
    
    if save == 's':
        env_path = Path(".env")
        
        # Ler .env existente se houver
        env_content = ""
        if env_path.exists():
            env_content = env_path.read_text(encoding='utf-8')
        
        # Atualizar ou adicionar variáveis
        lines = env_content.split('\n') if env_content else []
        new_lines = []
        jwt_found = False
        api_key_found = False
        
        for line in lines:
            if line.startswith('JWT_SECRET_KEY='):
                new_lines.append(f'JWT_SECRET_KEY={jwt_secret}')
                jwt_found = True
            elif line.startswith('API_KEY_1='):
                new_lines.append(f'API_KEY_1={api_key}')
                api_key_found = True
            else:
                new_lines.append(line)
        
        if not jwt_found:
            new_lines.append(f'JWT_SECRET_KEY={jwt_secret}')
        if not api_key_found:
            new_lines.append(f'API_KEY_1={api_key}')
        
        # Salvar
        env_path.write_text('\n'.join(new_lines), encoding='utf-8')
        print(f"\n✅ Chaves salvas em {env_path.absolute()}")
    else:
        print("\n⚠️  Lembre-se de adicionar essas chaves ao seu arquivo .env manualmente!")
    
    print()


if __name__ == "__main__":
    main()

