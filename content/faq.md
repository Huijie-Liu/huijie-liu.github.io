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
  <input type="hidden" id="threadId">
</div>

<script>
    document.getElementById('userInput').addEventListener('keypress', function(event) {
        if (event.keyCode === 13) { // 13 是回车键的键码
            event.preventDefault(); // 阻止默认行为（防止表单提交等）
            ask(); // 调用发送消息的函数
        }
    });

    async function ask() {
        const responses = document.getElementById('responses');
        const message = document.getElementById('userInput').value;
        const threadIdInput = document.getElementById('threadId'); // 假设有一个隐藏的输入框用于存储 threadId

        if (!message.trim()) return;
        loadingIndicator.style.display = 'block';

        try {
            // 准备请求体，如果有 threadId，则一起发送
            const requestBody = { message: message };
            if (threadIdInput.value) {
                requestBody.threadId = threadIdInput.value;
            }

            // 发送请求到 Cloud Function
            const response = await fetch('https://us-central1-jay-cloud-405910.cloudfunctions.net/openai_assistant', {
                method: 'POST',
                headers: {
                    'Accept': '*/*', // 期望接收任何类型的响应
                    'Accept-Encoding': 'gzip, deflate, br', // 可接受的内容编码
                    'Connection': 'keep-alive', // 保持长连接
                    'Content-Type': 'application/json', // 发送的数据类型为 JSON
                    'User-Agent': 'Mozilla/5.0' // 设置 User-Agent
                },
                body: JSON.stringify(requestBody)
            });

            // 解析响应数据
            const data = await response.json();
            const assistantMessage = data.message;
            document.getElementById('threadId').value = data.thread_id;

            if (assistantMessage) {
                responses.innerHTML += `<div class="message user-message"><strong>You:</strong> ${message}</div>`;
                responses.innerHTML += `<div class="message assistant-message"><strong>Assistant:</strong> ${assistantMessage}</div>`;
            } else {
                console.error('No assistant message found');
            }

            // 如果响应中包含 threadId，则保存它以供后续使用
            if (data.threadId) {
                threadIdInput.value = data.threadId;
            }
        } catch (error) {
            console.error('Error:', error);
        }

        document.getElementById('userInput').value = ''; // 清空输入框
        responses.scrollTop = responses.scrollHeight; // 滚动到底部
        loadingIndicator.style.display = 'none'; // 隐藏加载提示
    }
</script>
