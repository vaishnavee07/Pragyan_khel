const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files
app.use(express.static(__dirname));

// Main route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, () => {
    console.log('='.repeat(60));
    console.log('🎥 SENTRAVISION WEB SERVER STARTED');
    console.log('='.repeat(60));
    console.log(`📡 Server running at: http://localhost:${PORT}`);
    console.log(`🌐 Network access: http://[YOUR-IP]:${PORT}`);
    console.log('='.repeat(60));
    console.log('📋 INSTRUCTIONS:');
    console.log('1. Open the URL above in your browser');
    console.log('2. Allow camera permissions when prompted');
    console.log('3. AI models will load automatically');
    console.log('4. Real-time detection will start');
    console.log('='.repeat(60));
    console.log('Press Ctrl+C to stop the server');
    console.log('='.repeat(60));
});
