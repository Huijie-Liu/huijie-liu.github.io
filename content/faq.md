---
title: "FAQ"
# layout: "archives"
url: "/faq"
summary: "faq"
ShowToc: false
---

# Chat with Me

<style>
  #chat {
    background-color: #f4f4f4;
    max-width: 100%;
    margin: 20px auto;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  #userInput {
    width: calc(100% - 100px);
    padding: 10px;
    margin-right: 10px;
    border: 1px solid #ddd;
    border-radius: 5px;
  }

  #sendButton {
    padding: 10px 15px;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
  }

  #sendButton:hover {
    background-color: #0056b3;
  }

  .message {
    margin-bottom: 10px;
    padding: 10px;
    border-radius: 5px;
  }

  .user-message {
    background-color: #007bff;
    color: white;
    text-align: right;
  }

  .assistant-message {
    background-color: #e9ecef;
    color: #333;
  }
</style>

<div id="chat">
  <div id="responses"></div>
  <div id="loadingIndicator" style="display: none;">Loading...</div>
  <input type="text" id="userInput" placeholder="Ask a question">
  <button id="sendButton" onclick="ask()">Send</button>
</div>

<script>
    document.getElementById('userInput').addEventListener('keypress', function(event) {
        if (event.keyCode === 13) { // 13 是回车键的键码
            event.preventDefault(); // 阻止默认行为（防止表单提交等）
            ask(); // 调用发送消息的函数
        }
    });

    const openaiApiKey = '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';
    // const openaiApiKey = decrypt("vn0FqvOKF;4nm:kE[LlgZ8:W6EoenIMDerp<Dosp]o\X9qRekZf");
    const assistantId = 'asst_fQYHhCKqDBguL9bQOVJ6zOHc';

    async function createThread() {
        const threadResponse = await fetch('https://api.openai.com/v1/threads', {
                method: 'POST',
                headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${openaiApiKey}`,
                'OpenAI-Beta': 'assistants=v1'
                }
            });

        const threadData = await threadResponse.json();
        return threadData.id;
    }

    async function addMessageToThread(threadId, message) {
        const response = await fetch(`https://api.openai.com/v1/threads/${threadId}/messages`, {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${openaiApiKey}`,
            'OpenAI-Beta': 'assistants=v1'
            },
            body: JSON.stringify({
            role: 'user',
            content: message
            })
        });

        const data = await response.json();
        return data;
    }

    async function createRun(threadId) {
        const response = await fetch(`https://api.openai.com/v1/threads/${threadId}/runs`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${openaiApiKey}`,
                'Content-Type': 'application/json',
                'OpenAI-Beta': 'assistants=v1'
            },
            body: JSON.stringify({
                assistant_id: assistantId
            })
        });

        const data = await response.json();
        return data.id;
    }

    async function getMessages(threadId) {
        const response = await fetch(`https://api.openai.com/v1/threads/${threadId}/messages`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${openaiApiKey}`,
                'OpenAI-Beta': 'assistants=v1'
            }
        });

        const data = await response.json();
        return data.data;
    }

    async function ask() {
        const responses = document.getElementById('responses');
        const message = document.getElementById('userInput').value;

        if (!message.trim()) return;
        loadingIndicator.style.display = 'block';

        const threadId = await createThread();
        // console.log('Thread ID:', threadId);

        const messageResponse = await addMessageToThread(threadId, message);
        // console.log('Response:', messageResponse)

        const runId = await createRun(threadId);
        // console.log('Run ID:', runId);

        // Wait for the run to complete and get the assistant's messages
        setTimeout(async () => {
            const messages = await getMessages(threadId);
            // console.log('messages', messages)
            if (Array.isArray(messages) && messages.length > 0) {
                const assistantMessage = messages.find(msg => msg.role === 'assistant')?.content[0]?.text?.value;;
                if (assistantMessage) {
                    responses.innerHTML += `<div class="message user-message"><strong>You:</strong> ${message}</div>`;
                    responses.innerHTML += `<div class="message assistant-message"><strong>Assistant:</strong> ${assistantMessage}</div>`;
                } else {
                    console.error('No assistant message found');
                }
            } else {
                console.error('No messages returned from the thread');
            }
            document.getElementById('userInput').value = ''; // 清空输入框
            responses.scrollTop = responses.scrollHeight; // 滚动到底部
            loadingIndicator.style.display = 'none'; // 隐藏加载提示
        }, 5000); // Adjust the timeout as needed
    }

function decrypt(c) { return c.split('').map(ch => String.fromCharCode(((ch.charCodeAt(0) - 3) % 256) * 2 / 2 + 1 / 2)).join(''); }
</script>
