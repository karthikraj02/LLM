document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const chatHistory = document.getElementById('chat-history');
    const greetingArea = document.getElementById('greeting-area');
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const navSettings = document.getElementById('nav-settings');
    const navChat = document.getElementById('nav-chat');
    const settingsPanel = document.getElementById('settings-panel');

    // Settings
    const tempSlider = document.getElementById('temp-slider');
    const tokensSlider = document.getElementById('tokens-slider');
    const topkSlider = document.getElementById('topk-slider');
    const toppSlider = document.getElementById('topp-slider');

    const tempVal = document.getElementById('temp-val');
    const tokensVal = document.getElementById('tokens-val');
    const topkVal = document.getElementById('topk-val');
    const toppVal = document.getElementById('topp-val');

    // Update value displays
    tempSlider.oninput = () => tempVal.textContent = tempSlider.value;
    tokensSlider.oninput = () => tokensVal.textContent = tokensSlider.value;
    topkSlider.oninput = () => topkVal.textContent = topkSlider.value;
    toppSlider.oninput = () => toppVal.textContent = toppSlider.value;

    // Sidebar toggle
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });

    // Settings nav button toggles settings panel
    navSettings.addEventListener('click', () => {
        const isVisible = settingsPanel.style.display !== 'none';
        settingsPanel.style.display = isVisible ? 'none' : 'block';
        navSettings.classList.toggle('active', !isVisible);
        navChat.classList.toggle('active', isVisible);
    });

    navChat.addEventListener('click', () => {
        settingsPanel.style.display = 'none';
        navSettings.classList.remove('active');
        navChat.classList.add('active');
    });

    // Auto-resize textarea
    promptInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Send message on Enter (but allow Shift+Enter for newline)
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    async function sendMessage() {
        const text = promptInput.value.trim();
        if (!text) return;

        // Hide greeting, show chat history
        greetingArea.style.display = 'none';
        chatHistory.style.display = 'flex';

        // Add user message to UI
        appendMessage('user', text);

        // Clear input and reset height
        promptInput.value = '';
        promptInput.style.height = 'auto';

        // Disable input while waiting
        promptInput.disabled = true;
        sendBtn.disabled = true;

        // Add loading indicator
        const loadingId = appendLoading();

        try {
            const response = await fetch('http://localhost:8000/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: text,
                    temperature: parseFloat(tempSlider.value),
                    max_tokens: parseInt(tokensSlider.value),
                    top_k: parseInt(topkSlider.value),
                    top_p: parseFloat(toppSlider.value)
                })
            });

            removeLoading(loadingId);

            if (!response.ok) {
                const err = await response.json();
                appendMessage('system', `Error: ${err.detail || 'Failed to connect to backend.'}`);
                return;
            }

            const data = await response.json();

            // Simulate streaming for better UX
            await simulateStream(data.generated_text);

        } catch (error) {
            removeLoading(loadingId);
            appendMessage('system', 'Error connecting to the model server. Is it running?');
        } finally {
            promptInput.disabled = false;
            sendBtn.disabled = false;
            promptInput.focus();
        }
    }

    function appendMessage(role, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}-message`;

        const avatarContent = role === 'user' ? 'K' : role === 'system' ? '!' : '✦';

        msgDiv.innerHTML = `
            <div class="avatar">${avatarContent}</div>
            <div class="content"><p>${escapeHtml(text)}</p></div>
        `;

        chatHistory.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    function appendLoading() {
        const id = 'loading-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot-message';
        msgDiv.id = id;

        msgDiv.innerHTML = `
            <div class="avatar">✦</div>
            <div class="content">
                <div class="typing-indicator">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        `;

        chatHistory.appendChild(msgDiv);
        scrollToBottom();
        return id;
    }

    function removeLoading(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    async function simulateStream(fullText) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot-message';

        const p = document.createElement('p');

        msgDiv.innerHTML = `<div class="avatar">✦</div>`;
        const contentDiv = document.createElement('div');
        contentDiv.className = 'content';
        contentDiv.appendChild(p);
        msgDiv.appendChild(contentDiv);

        chatHistory.appendChild(msgDiv);

        // Simulating the streaming of words
        const words = fullText.split(' ');
        let currentText = '';

        for (let i = 0; i < words.length; i++) {
            currentText += words[i] + (i < words.length - 1 ? ' ' : '');
            p.innerHTML = escapeHtml(currentText);
            scrollToBottom();

            // Wait slightly between words
            await new Promise(r => setTimeout(r, 20 + Math.random() * 40));
        }
    }

    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function escapeHtml(unsafe) {
        return unsafe.toString()
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;")
            .replace(/\n/g, "<br>");
    }
});

