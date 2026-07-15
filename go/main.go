package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const apiURL = "https://openrouter.ai/api/v1/chat/completions"

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

type ChatResponse struct {
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
}

func complete(client *http.Client, apiKey, model string, messages []Message) (string, error) {
	payload, err := json.Marshal(ChatRequest{Model: model, Messages: messages})
	if err != nil {
		return "", fmt.Errorf("encode request: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, apiURL, bytes.NewReader(payload))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("call OpenRouter: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("OpenRouter returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}

	var result ChatResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("OpenRouter returned no choices")
	}

	return result.Choices[0].Message.Content, nil
}

func loadDotEnv() {
	data, err := os.ReadFile("../.env")
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		if _, exists := os.LookupEnv(key); !exists {
			os.Setenv(key, val)
		}
	}
}

func main() {
	loadDotEnv()
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "Set OPENROUTER_API_KEY before running this program.")
		os.Exit(1)
	}

	model := os.Getenv("OPENROUTER_MODEL")
	if model == "" {
		model = "openrouter/auto"
	}

	messages := []Message{{Role: "system", Content: "You are a helpful, concise assistant."}}
	client := &http.Client{Timeout: 2 * time.Minute}
	scanner := bufio.NewScanner(os.Stdin)

	fmt.Printf("OpenRouter chat agent using %s. Type 'exit' to quit.\n", model)

	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			fmt.Println()
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if strings.EqualFold(input, "exit") || strings.EqualFold(input, "quit") {
			break
		}

		messages = append(messages, Message{Role: "user", Content: input})
		answer, err := complete(client, apiKey, model, messages)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Error:", err)
			continue
		}

		fmt.Printf("Assistant: %s\n\n", answer)
		messages = append(messages, Message{Role: "assistant", Content: answer})
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "Input error:", err)
		os.Exit(1)
	}
}
