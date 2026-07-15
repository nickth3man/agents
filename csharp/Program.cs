using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

internal static class Program
{
    private const string ApiUrl = "https://openrouter.ai/api/v1/chat/completions";

    private static async Task<int> Main()
    {
        LoadDotEnv();
        string? apiKey = Environment.GetEnvironmentVariable("OPENROUTER_API_KEY");
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            Console.Error.WriteLine("Set OPENROUTER_API_KEY before running this program.");
            return 1;
        }

        string model = Environment.GetEnvironmentVariable("OPENROUTER_MODEL")
            ?? "openrouter/auto";

        using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
        client.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", apiKey);

        var messages = new List<Message>
        {
            new("system", "You are a helpful, concise assistant.")
        };

        Console.WriteLine($"OpenRouter chat agent using {model}. Type 'exit' to quit.");

        while (true)
        {
            Console.Write("You: ");
            string? input = Console.ReadLine();
            if (input is null)
            {
                Console.WriteLine();
                break;
            }

            input = input.Trim();
            if (input.Length == 0)
            {
                continue;
            }
            if (input.Equals("exit", StringComparison.OrdinalIgnoreCase) ||
                input.Equals("quit", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            messages.Add(new Message("user", input));

            try
            {
                string answer = await CompleteAsync(client, model, messages);
                Console.WriteLine($"Assistant: {answer}\n");
                messages.Add(new Message("assistant", answer));
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error: {ex.Message}");
            }
        }

        return 0;
    }

    private static async Task<string> CompleteAsync(
        HttpClient client,
        string model,
        List<Message> messages)
    {
        var payload = new ChatRequest(model, messages);
        string json = JsonSerializer.Serialize(payload);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        using HttpResponseMessage response = await client.PostAsync(ApiUrl, content);
        string body = await response.Content.ReadAsStringAsync();

        if (!response.IsSuccessStatusCode)
        {
            throw new HttpRequestException(
                $"OpenRouter returned {(int)response.StatusCode} {response.ReasonPhrase}: {body}");
        }

        ChatResponse? result = JsonSerializer.Deserialize<ChatResponse>(body);
        string? answer = result?.Choices?.FirstOrDefault()?.Message?.Content;

        return answer ?? throw new InvalidOperationException(
            $"OpenRouter returned an unexpected response: {body}");
    }

    private static void LoadDotEnv()
    {
        string path = Path.Combine(Directory.GetCurrentDirectory(), "..", ".env");
        if (!File.Exists(path)) return;
        foreach (string line in File.ReadAllLines(path))
        {
            string trimmed = line.Trim();
            if (trimmed.Length == 0 || trimmed.StartsWith('#')) continue;
            int eq = trimmed.IndexOf('=');
            if (eq < 0) continue;
            string key = trimmed[..eq].Trim();
            string value = trimmed[(eq + 1)..].Trim();
            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable(key)))
            {
                Environment.SetEnvironmentVariable(key, value);
            }
        }
    }
}

internal sealed record Message(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string Content);

internal sealed record ChatRequest(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("messages")] List<Message> Messages);

internal sealed class ChatResponse
{
    [JsonPropertyName("choices")]
    public List<Choice>? Choices { get; init; }
}

internal sealed class Choice
{
    [JsonPropertyName("message")]
    public Message? Message { get; init; }
}
