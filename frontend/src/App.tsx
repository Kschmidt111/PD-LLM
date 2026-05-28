import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import "./App.css";

type Citation = {
  source_id?: string;
  title?: string;
  chunk_id?: string;
  url?: string;
};

type ChatResponse = {
  answer?: string;
  response?: string;
  citations?: Citation[];
};

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const apiBase = useMemo(() => {
    const base = import.meta.env.VITE_API_BASE_URL as string | undefined;
    return base?.trim() || DEFAULT_API_BASE;
  }, []);

  const submitQuestion = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const prompt = question.trim();
    if (!prompt) {
      return;
    }

    setIsLoading(true);
    setError("");
    setAnswer("");
    setCitations([]);

    try {
      const response = await fetch(`${apiBase}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: prompt }),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data = (await response.json()) as ChatResponse;
      setAnswer(data.answer ?? data.response ?? "No answer was returned.");
      setCitations(data.citations ?? []);
    } catch (caughtError) {
      const message =
        caughtError instanceof Error ? caughtError.message : "Unknown request failure";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <header className="app-header">
        <h1>PD Research Assistant</h1>
        <p className="subtitle">Research and education only. Not for diagnosis or personal risk advice.</p>
      </header>

      <section className="chat-panel">
        <form onSubmit={submitQuestion} className="question-form">
          <label htmlFor="question">Ask a research question</label>
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Example: What evidence supports alpha-synuclein as a PD biomarker?"
            rows={5}
          />
          <button type="submit" disabled={isLoading}>
            {isLoading ? "Asking..." : "Ask"}
          </button>
        </form>

        <article className="answer-panel">
          <h2>Answer</h2>
          {error ? <p className="error">Error: {error}</p> : null}
          {!error && !answer ? <p className="placeholder">Submit a question to see grounded output.</p> : null}
          {answer ? <p className="answer">{answer}</p> : null}
        </article>

        <aside className="citations-panel">
          <h2>Citations</h2>
          {citations.length === 0 ? (
            <p className="placeholder">No citations returned yet.</p>
          ) : (
            <ul>
              {citations.map((citation, index) => (
                <li key={`${citation.source_id ?? "source"}-${index}`}>
                  <div><strong>Source:</strong> {citation.source_id ?? "n/a"}</div>
                  <div><strong>Title:</strong> {citation.title ?? "n/a"}</div>
                  <div><strong>Chunk:</strong> {citation.chunk_id ?? "n/a"}</div>
                  <div>
                    <strong>URL:</strong>{" "}
                    {citation.url ? (
                      <a href={citation.url} target="_blank" rel="noreferrer">
                        {citation.url}
                      </a>
                    ) : (
                      "n/a"
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </aside>
      </section>
    </main>
  );
}

export default App;
