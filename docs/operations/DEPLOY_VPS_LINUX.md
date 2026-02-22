# Despliegue en VPS Linux (AWS Dublin)

Guia para desplegar helios-temperature en Ubuntu 22.04 LTS con Python 3.12.

---

## 1. Requisitos previos

- VPS Ubuntu 22.04 LTS (AWS EC2, region `eu-west-1` Dublin)
- Acceso SSH con key pair
- Security Group configurado (ver seccion 7)

## 2. Instalar Python 3.12

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
python3.12 --version
```

## 3. Instalar dependencias del sistema

```bash
sudo apt install -y git nginx
```

## 4. Clonar repositorio y configurar entorno

```bash
cd /opt
sudo mkdir helios && sudo chown $USER:$USER helios
git clone <URL_DEL_REPO> /opt/helios
cd /opt/helios

python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Configurar variables de entorno

Crear `/opt/helios/.env` con las claves necesarias:

```env
# Gemini API (obligatorio)
GOOGLE_API_KEY=<tu_clave_gemini>

# Synoptic Data API - Personal Weather Stations (obligatorio)
SYNOPTIC_API_TOKEN=<tu_token_synoptic>

# Weather.com / Wunderground (opcional, recomendado si usas WU como fuente PWS)
WUNDERGROUND_API_KEY=<tu_api_key_weather_com>
WUNDERGROUND_PWS_ENABLED=true
WUNDERGROUND_PWS_REGISTRY_PATH=data/wu_pws_station_registry.json

# Polymarket (obligatorio)
POLYMARKET_PRIVATE_KEY=<tu_private_key>
POLYMARKET_API_KEY=<tu_api_key>
POLYMARKET_API_SECRET=<tu_api_secret>
POLYMARKET_API_PASSPHRASE=<tu_passphrase>

# Debug (opcional)
POLYMARKET_WS_DEBUG=0
```

> **Importante**: No commitear `.env` al repositorio. Asegurar permisos restrictivos:
> ```bash
> chmod 600 /opt/helios/.env
> ```

## 6. Crear directorios runtime

```bash
mkdir -p /opt/helios/logs
mkdir -p /opt/helios/data
```

La base de datos SQLite (`helios_weather.db`) se crea automaticamente al arrancar.

## 7. Servicio systemd

Crear `/etc/systemd/system/helios.service`:

```bash
sudo tee /etc/systemd/system/helios.service > /dev/null << 'EOF'
[Unit]
Description=Helios Temperature Trading System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/helios
Environment=PATH=/opt/helios/venv/bin:/usr/bin
ExecStart=/opt/helios/venv/bin/python web_server.py
Restart=always
RestartSec=5
StandardOutput=append:/opt/helios/logs/helios.log
StandardError=append:/opt/helios/logs/helios-error.log

[Install]
WantedBy=multi-user.target
EOF
```

Activar y arrancar:

```bash
sudo systemctl daemon-reload
sudo systemctl enable helios
sudo systemctl start helios
sudo systemctl status helios
```

Verificar que esta corriendo:

```bash
curl http://localhost:8000
```

### Comandos utiles

```bash
# Ver logs en tiempo real
sudo journalctl -u helios -f

# Reiniciar
sudo systemctl restart helios

# Ver logs de aplicacion
tail -f /opt/helios/logs/helios.log
tail -f /opt/helios/logs/helios-error.log
```

## 7.1 (Opcional) Auto-refresco WU PWS (systemd timer)

Si usas Wunderground PWS como fuente adicional (para `WUNDERGROUND` en el panel PWS), conviene refrescar
periodicamente la lista de estaciones cercanas validas para todas las estaciones activas
(`KLGA,KATL,KORD,KMIA,KDAL,LFPG,EGLC,LTAC`).

En el repo hay templates en `deploy/systemd/`:

```bash
sudo cp /opt/helios/deploy/systemd/helios-wu-discover.service /etc/systemd/system/
sudo cp /opt/helios/deploy/systemd/helios-wu-discover.timer /etc/systemd/system/

sudo systemctl daemon-reload

# Ejecutar una vez ahora (opcional)
sudo systemctl start helios-wu-discover.service

# Habilitar timer en boot
sudo systemctl enable --now helios-wu-discover.timer

sudo systemctl status helios-wu-discover.timer --no-pager
sudo systemctl list-timers --all | grep wu || true

tail -f /opt/helios/logs/wu_discover.log
tail -f /opt/helios/logs/wu_discover_error.log
```

## 8. Nginx reverse proxy

Crear `/etc/nginx/sites-available/helios`:

```bash
sudo tee /etc/nginx/sites-available/helios > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400s;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF
```

Activar y reiniciar Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/helios /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

## 9. AWS Security Group

En la consola de AWS EC2, configurar el Security Group de la instancia:

| Tipo | Protocolo | Puerto | Origen | Uso |
|------|-----------|--------|--------|-----|
| SSH | TCP | 22 | Tu IP / 0.0.0.0/0 | Acceso SSH |
| HTTP | TCP | 80 | 0.0.0.0/0 | Web publica |
| HTTPS | TCP | 443 | 0.0.0.0/0 | Web publica (futuro SSL) |

> No exponer el puerto 8000 directamente. Nginx hace de proxy en el 80.

## 10. Verificacion post-despliegue

```bash
# 1. Servicio activo
sudo systemctl status helios
# Debe mostrar: active (running)

# 2. Respuesta local
curl -s http://localhost:8000 | head -5
# Debe devolver HTML

# 3. Respuesta via Nginx
curl -s http://localhost | head -5
# Debe devolver HTML

# 4. Respuesta externa (desde tu maquina local)
curl http://<IP_PUBLICA_EC2>
# Debe devolver HTML
```

La IP publica se encuentra en la consola EC2 > Instances > Public IPv4 address.

## 11. SSL con Let's Encrypt (opcional)

Si tienes un dominio apuntando a la IP:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d tudominio.com
```

Certbot configura Nginx automaticamente y renueva el certificado cada 90 dias.

---

## Resumen de rutas

| Ruta | Contenido |
|------|-----------|
| `/opt/helios/` | Codigo fuente |
| `/opt/helios/venv/` | Entorno virtual Python |
| `/opt/helios/.env` | Variables de entorno |
| `/opt/helios/helios_weather.db` | Base de datos SQLite |
| `/opt/helios/logs/` | Logs de aplicacion |
| `/opt/helios/data/` | Datos runtime |
| `/etc/systemd/system/helios.service` | Servicio systemd |
| `/etc/nginx/sites-available/helios` | Config Nginx |
