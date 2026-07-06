const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 3200;
const CRM_PATH = path.resolve(__dirname, '../local_crm.json');

const server = http.createServer((req, res) => {
    if (req.url === '/' || req.url === '/index.html') {
        const filePath = path.join(__dirname, 'index.html');
        fs.readFile(filePath, 'utf8', (err, data) => {
            if (err) {
                res.writeHead(500);
                res.end('Error loading the page');
                return;
            }
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(data);
        });
    } else if (req.url === '/local_crm.json') {
        fs.readFile(CRM_PATH, 'utf8', (err, data) => {
            if (err) {
                res.writeHead(500);
                res.end(JSON.stringify({ error: 'CRM data not found. Run generate-crm.py --export-json first.' }));
                return;
            }
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(data);
        });
    } else {
        res.writeHead(404);
        res.end('Not found');
    }
});

server.listen(PORT, () => {
    console.log(`Customer Data Viewer running at http://localhost:${PORT}`);
    console.log(`Open in your browser to view customer data`);
});