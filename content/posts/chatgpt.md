+++
title = 'Chatgpt'
date = 2024-04-02T18:45:02+08:00
draft = true
+++

## Chat with Assistant

<div id="chat">
  <input type="text" id="userInput">
  <button onclick="sendMessage()">Send</button>
  <div id="responses"></div>
</div>

<script>
  async function sendMessage() {
    const input = document.getElementById('userInput');
    const responses = document.getElementById('responses');

    const response = await fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ messages: [{ role: 'user', content: input.value }] }),
    });

    const data = await response.json();
    const reply = data.choices[0].message.content;

    responses.innerHTML += `<div><strong>You:</strong> ${input.value}</div>`;
    responses.innerHTML += `<div><strong>Assistant:</strong> ${reply}</div>`;

    input.value = ''; // Clear input field
  }
</script>
