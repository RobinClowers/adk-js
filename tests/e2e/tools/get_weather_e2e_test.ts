/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  FunctionTool,
  InMemoryRunner,
  LlmAgent,
  StreamingMode,
} from '@google/adk';
import {createUserContent} from '@google/genai';
import * as dotenv from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';
import {describe, expect, it} from 'vitest';
import {z} from 'zod';

const envPath = path.resolve(__dirname, '.env');
const envExists = fs.existsSync(envPath);

if (envExists) {
  dotenv.config({path: envPath});
}

// Check for API keys to skip tests if not available
const hasAKey =
  !!process.env.GEMINI_API_KEY ||
  !!process.env.GOOGLE_GENAI_API_KEY ||
  !!process.env.GOOGLE_CLOUD_PROJECT;

describe.skipIf(!hasAKey)('E2E Weather Tool Test', () => {
  it('produces text after each tool call across multiple turns', async () => {
    // The text finalization bug only manifests reliably across multiple turns:
    // after some conversation history with prior function calls, thinking
    // models start streaming an empty text chunk with finishReason STOP after
    // the functionCall chunk. Without the fix that empty chunk is treated as
    // a final response, so the agent stops before the follow-up model call
    // that would produce the actual text summary. A single-turn test often
    // passes even without the fix, so we run several turns and assert that
    // every turn produces non-empty text.
    const toolCalls: string[] = [];
    const getWeather = new FunctionTool({
      name: 'get_weather',
      description: 'Get current weather for a city',
      parameters: z.object({city: z.string()}),
      execute: async ({city}: {city: string}) => {
        toolCalls.push(city);
        return {city, temp: '72F', condition: 'sunny'};
      },
    });

    const agent = new LlmAgent({
      name: 'test_agent',
      model: 'gemini-3-flash-preview',
      instruction:
        'You help with weather questions. Always use the get_weather tool, then summarize the result in a short sentence.',
      tools: [getWeather],
    });

    const runner = new InMemoryRunner({agent, appName: 'weather_test_app'});
    const session = await runner.sessionService.createSession({
      appName: 'weather_test_app',
      userId: 'test_user',
    });

    const prompts = [
      'What is the weather in Seattle?',
      'What about Tokyo?',
      'And London?',
      'How about Paris?',
    ];

    const turnResults: Array<{prompt: string; text: string}> = [];

    for (const prompt of prompts) {
      let responseText = '';
      for await (const event of runner.runAsync({
        userId: 'test_user',
        sessionId: session.id,
        newMessage: createUserContent(prompt),
        // The bug only manifests on the streaming path. Default is NONE.
        runConfig: {streamingMode: StreamingMode.SSE},
      })) {
        if (event.author !== 'test_agent' || !event.content?.parts) continue;
        for (const part of event.content.parts) {
          // Skip thought parts; only accumulate user-visible text.
          if (part.thought) continue;
          if (part.text) responseText += part.text;
        }
      }
      turnResults.push({prompt, text: responseText});
    }

    // The tool must be invoked on every turn.
    expect(toolCalls.length).toBe(prompts.length);

    // Every turn must produce non-empty text. Without the fix, at least one
    // of the later turns terminates after the tool response with no text.
    for (const {prompt, text} of turnResults) {
      expect(text.trim(), `no text produced for prompt "${prompt}"`).not.toBe(
        '',
      );
    }
  }, 120000);
});
