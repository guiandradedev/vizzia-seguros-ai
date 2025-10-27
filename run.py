from src.app.main import create_app
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = create_app()

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = os.getenv('PORT') or os.getenv('APP_PORT') or os.getenv('FLASK_RUN_PORT') or 5000
    try:
        port = int(port)
    except Exception:
        port = 5000

    debug = os.getenv('FLASK_DEBUG', 'True').lower() in ('1', 'true', 'yes')

    app.run(host=host, port=port, debug=debug)
