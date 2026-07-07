const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3300;
const BACKEND_URL = 'http://localhost:8050';
const FRONTEND_DIR = path.resolve(__dirname);

const server = http.createServer((req, res) => {
    // Add CORS headers to all responses
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    
    // Handle preflight
    if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
    }
    
    // Route handling
    let filePath;
    
    // Proxy SSE endpoint to backend
    if (req.url === '/admin/trace') {
        const proxyReq = http.request({
            hostname: 'localhost',
            port: 8050,
            path: '/admin/trace',
            method: 'GET',
            headers: {
                'Host': 'localhost:8050',
                'Accept': 'text/event-stream'
            }
        }, (proxyRes) => {
            // Copy headers from backend to client
            res.writeHead(proxyRes.statusCode, {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*' // Add CORS headers
            });
            
            proxyRes.on('data', (chunk) => {
                res.write(chunk);
            });
            
            proxyRes.on('end', () => {
                res.end();
            });
        });
        
        proxyReq.on('error', (err) => {
            console.error('Proxy error:', err);
            res.writeHead(502);
            res.end('Bad Gateway');
        });
        
        proxyReq.end();
        return;
    }
    if (req.url === '/' || req.url === '/index.html') {
        filePath = path.join(FRONTEND_DIR, 'index.html');
    } else if (req.url === '/admin.html') {
        filePath = path.join(FRONTEND_DIR, 'admin.html');
    } else {
        // Try to serve the requested file
        const fullPath = path.join(FRONTEND_DIR, req.url);
        if (fullPath.startsWith(FRONTEND_DIR) && fs.existsSync(fullPath)) {
            filePath = fullPath;
        } else {
            // Fallback to index.html for SPA behavior
            filePath = path.join(FRONTEND_DIR, 'index.html');
        }
    }
    
    // Check if file exists
    fs.access(filePath, fs.constants.R_OK, (err) => {
        if (err) {
            res.writeHead(404);
            res.end('File not found');
            return;
        }
        
        // Determine content type
        const ext = path.extname(filePath).toLowerCase();
        const contentType = {
            '.html': 'text/html',
            '.js': 'application/javascript',
            '.css': 'text/css',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.svg': 'image/svg+xml'
        }[ext] || 'text/plain';
        
        fs.readFile(filePath, 'utf8', (err, data) => {
            if (err) {
                res.writeHead(500);
                res.end('Error loading file');
                return;
            }
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(data);
        });
    });
});

server.listen(PORT, () => {
    console.log(`Support Agent Frontend running at http://localhost:${PORT}`);
    console.log(`  - Customer Chat: http://localhost:${PORT}/`);
    console.log(`  - Admin Console: http://localhost:${PORT}/admin.html`);
});