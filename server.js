const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const NIM_API_BASE = 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

const DEFAULT_MODEL = 'nvidia/nemotron-3-ultra-550b-a55b';

const MODEL_MAPPING = {
  'gpt-3.5-turbo':  'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-4':          'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-4-turbo':    'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-4o':         'nvidia/nemotron-3-ultra-550b-a55b',
  'claude-3-opus':  'nvidia/nemotron-3-ultra-550b-a55b',
  'claude-3-sonnet':'nvidia/nemotron-3-ultra-550b-a55b',
  'gemini-pro':     'nvidia/nemotron-3-ultra-550b-a55b',
};

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI → NVIDIA NIM Proxy',
    default_model: DEFAULT_MODEL,
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
  });
});

// Models list
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy',
  }));
  res.json({ object: 'list', data: models });
});

// Chat completions
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    const nimModel = MODEL_MAPPING[model] || DEFAULT_MODEL;

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature || 0.6,
      max_tokens: Math.max(max_tokens || 16384, 4096),
      stream: stream || false,
    };

    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }

    let response, lastErr;
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        response = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          nimRequest,
          {
            headers: {
              Authorization: `Bearer ${NIM_API_KEY}`,
              'Content-Type': 'application/json',
            },
            responseType: stream ? 'stream' : 'json',
          }
        );
        break;
      } catch (e) {
        lastErr = e;
        if (e.response?.status === 503 && attempt < 3) {
          await new Promise(r => setTimeout(r, 2000 * attempt));
        } else { throw e; }
      }
    }

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningOpen = false;

      response.data.on('data', chunk => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) { res.write(line + '\n'); continue; }
          if (line.includes('[DONE]')) { res.write(line + '\n'); continue; }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;
            if (delta) {
              const reasoning = delta.reasoning_content;
              const content   = delta.content;

              if (SHOW_REASONING) {
                let out = '';
                if (reasoning && !reasoningOpen) { out = '<think>\n' + reasoning; reasoningOpen = true; }
                else if (reasoning) { out = reasoning; }
                if (content && reasoningOpen) { out += '</think>\n\n' + content; reasoningOpen = false; }
                else if (content) { out += content; }
                delta.content = out;
              } else {
                delta.content = content || '';
              }
              delete delta.reasoning_content;
            }
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch {
            res.write(line + '\n');
          }
        }
      });

      response.data.on('end', () => res.end());
      response.data.on('error', err => { console.error('Stream error:', err); res.end(); });

    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: response.data.choices.map(choice => {
          let content = choice.message?.content || '';
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            content = `<think>\n${choice.message.reasoning_content}\n</think>\n\n${content}`;
          }
          return {
            index: choice.index,
            message: { role: choice.message.role, content },
            finish_reason: choice.finish_reason,
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0,
        },
      };
      res.json(openaiResponse);
    }

  } catch (err) {
    console.error('Proxy error:', err.message);
    res.status(err.response?.status || 500).json({
      error: {
        message: err.message || 'Internal server error',
        type: 'invalid_request_error',
        code: err.response?.status || 500,
      },
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`Proxy running on port ${PORT}`);
  console.log(`Default model: ${DEFAULT_MODEL}`);
});
