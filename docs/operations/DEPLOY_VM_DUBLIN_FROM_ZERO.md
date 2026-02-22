# Despliegue Real Desde Cero (Windows -> VM Ubuntu 22.04 Dublin)

Guia paso a paso con el flujo real usado para `helios-temperature`, incluyendo:
- Git en Windows y en Ubuntu.
- Error de autenticacion `credential-manager-core`.
- Base de datos SQLite grande fuera de Git.
- Exposicion segura via `nginx` (80) manteniendo `8000` privado.
- Tunel SSH desde Windows para debug directo.

---

## 1) Antes de subir codigo: excluir base de datos y secretos

La base SQLite local no debe ir a GitHub. Ya hay reglas en `.gitignore` (`*.db`, `*.sqlite*`, etc.), pero si ya estaba trackeada hay que des-trackearla:

```powershell
git rm --cached helios_weather.db
git rm --cached .env
git add .gitignore
git commit -m "chore: ignore local db and env"
git push
```

Si quieres editar `.gitignore` en PowerShell, no uses heredoc estilo bash (`<<EOF`). Usa:

```powershell
Add-Content .gitignore "`n# Local runtime DB`n*.db`n*.sqlite*"
```

---

## 2) Conectarte a la VM (Ubuntu 22.04)

Desde Windows:

```powershell
ssh -i "C:\Users\anxoo\.ssh\LightsailDefaultKey-eu-west-1.pem" ubuntu@3.253.86.168
```

---

## 3) Preparar sistema en Ubuntu

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git nginx python3.12 python3.12-venv python3-pip
```

Si `python3.12` no existe en tu imagen:

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

---

## 4) Arreglar autenticacion Git en Ubuntu (caso `credential-manager-core`)

Si al clonar ves:

```text
git: 'credential-manager-core' is not a git command
```

limpia el helper heredado:

```bash
git config --global --unset-all credential.helper || true
```

### Opcion recomendada: GitHub CLI

```bash
type -p gh >/dev/null || (sudo apt update && sudo apt install -y gh)
gh auth login
gh auth setup-git
```

### Opcion alternativa: HTTPS + PAT

Al hacer `git clone`, usa usuario GitHub y como password pega un Personal Access Token.

---

## 5) Clonar repo en `/opt/helios`

```bash
cd /opt
sudo mkdir -p helios
sudo chown ubuntu:ubuntu helios
git clone https://github.com/anx00/helios-temperature.git /opt/helios
cd /opt/helios
```

---

## 6) Crear entorno e instalar dependencias

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7) Configurar `.env` en la VM

```bash
cp .env.example .env 2>/dev/null || touch .env
nano .env
chmod 600 .env
```

Variables minimas:

```env
GOOGLE_API_KEY=...
SYNOPTIC_API_TOKEN=...
POLYMARKET_PRIVATE_KEY=...
POLYMARKET_API_KEY=...
POLYMARKET_API_SECRET=...
POLYMARKET_API_PASSPHRASE=...
POLYMARKET_WS_DEBUG=0
```

Nunca subas `.env` al repo.

---

## 8) Crear carpetas runtime

```bash
mkdir -p /opt/helios/logs
mkdir -p /opt/helios/data
```

La DB SQLite se crea/actualiza localmente en la VM.

---

## 9) Probar app localmente en la VM (puerto 8000)

```bash
cd /opt/helios
source venv/bin/activate
python web_server.py
```

En otra shell de la VM:

```bash
curl -I http://127.0.0.1:8000
```

Si responde `200`, la app esta bien.

---

## 10) Publicar correctamente: `nginx` en 80 y app privada en 8000

Esto es lo esperado:
- `80` accesible desde fuera.
- `8000` cerrado desde fuera.

### Config nginx reverse proxy

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
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_buffering off;
        proxy_read_timeout 86400s;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/helios /etc/nginx/sites-enabled/ 2>/dev/null || true
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

En AWS Security Group:
- abrir `22` (SSH) solo tu IP si es posible.
- abrir `80` para web.
- no abrir `8000`.

---

## 11) Dejarlo 24/7 con `systemd`

```bash
sudo tee /etc/systemd/system/helios.service > /dev/null << 'EOF'
[Unit]
Description=Helios Temperature Web Server
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

sudo systemctl daemon-reload
sudo systemctl enable helios
sudo systemctl restart helios
sudo systemctl status helios --no-pager
```

Logs:

```bash
sudo journalctl -u helios -f
tail -f /opt/helios/logs/helios.log
```

---

## 12) Tunel SSH desde Windows (debug seguro sin abrir 8000)

Comando recomendado:

```powershell
ssh -i "C:\Users\anxoo\.ssh\LightsailDefaultKey-eu-west-1.pem" -N -T -L 8000:127.0.0.1:8000 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 ubuntu@3.253.86.168
```

Con el tunel activo, abre en tu PC:

```text
http://127.0.0.1:8000
```

---

## 13) Verificacion final

En VM:

```bash
curl -I http://127.0.0.1:8000
curl -I http://127.0.0.1
sudo systemctl status helios --no-pager
sudo systemctl status nginx --no-pager
```

En Windows:

```powershell
curl -I http://3.253.86.168
```

Esperado:
- responde `200` por `80` (nginx).
- la app funciona via dominio/IP publica.
- `8000` sigue sin exponerse publicamente.

---

## 14) Actualizar despliegue cuando hay cambios

```bash
cd /opt/helios
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart helios
sudo systemctl status helios --no-pager
```

