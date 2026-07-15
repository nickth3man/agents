use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error;
use std::io::{self, Write};
use std::time::Duration;

const API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

#[derive(Clone, Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

async fn complete(
    client: &Client,
    api_key: &str,
    model: &str,
    messages: &[Message],
) -> Result<String, Box<dyn Error>> {
    let response = client
        .post(API_URL)
        .bearer_auth(api_key)
        .json(&ChatRequest { model, messages })
        .send()
        .await?;

    let status = response.status();
    let body = response.text().await?;

    if !status.is_success() {
        return Err(format!("OpenRouter returned {status}: {body}").into());
    }

    let parsed: ChatResponse = serde_json::from_str(&body)?;
    let answer = parsed
        .choices
        .into_iter()
        .next()
        .map(|choice| choice.message.content)
        .ok_or("OpenRouter returned no choices")?;

    Ok(answer)
}

fn load_dotenv() {
    let path = std::path::Path::new("../.env");
    if !path.exists() {
        return;
    }
    if let Ok(contents) = std::fs::read_to_string(path) {
        for line in contents.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some(eq) = trimmed.find('=') {
                let key = trimmed[..eq].trim();
                let value = trimmed[eq + 1..].trim();
                if std::env::var(key).is_err() {
                    std::env::set_var(key, value);
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    load_dotenv();
    let api_key = match env::var("OPENROUTER_API_KEY") {
        Ok(value) if !value.trim().is_empty() => value,
        _ => {
            eprintln!("Set OPENROUTER_API_KEY before running this program.");
            std::process::exit(1);
        }
    };

    let model = env::var("OPENROUTER_MODEL").unwrap_or_else(|_| "openrouter/auto".to_string());
    let client = Client::builder().timeout(Duration::from_secs(120)).build()?;
    let mut messages = vec![Message {
        role: "system".to_string(),
        content: "You are a helpful, concise assistant.".to_string(),
    }];

    println!("OpenRouter chat agent using {model}. Type 'exit' to quit.");

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            println!();
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("quit") {
            break;
        }

        messages.push(Message {
            role: "user".to_string(),
            content: input.to_string(),
        });

        match complete(&client, &api_key, &model, &messages).await {
            Ok(answer) => {
                println!("Assistant: {answer}\n");
                messages.push(Message {
                    role: "assistant".to_string(),
                    content: answer,
                });
            }
            Err(error) => eprintln!("Error: {error}"),
        }
    }

    Ok(())
}
