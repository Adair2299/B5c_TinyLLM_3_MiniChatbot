
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const chatBox = document.getElementById("chat-box");

const tokenGrid = document.getElementById("token-grid");
const examplesList = document.getElementById("examples-list");

const randomTokensBtn = document.getElementById("random-tokens-btn");
const randomExamplesBtn = document.getElementById("random-examples-btn");

// ==========================================
// Chat functionality
// ==========================================

function addMessage(text, sender) {
    const message = document.createElement("div");

    message.classList.add("message", sender);
    message.textContent = text;

    chatBox.appendChild(message);
    chatBox.scrollTop = chatBox.scrollHeight;
}

chatForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const question = userInput.value.trim();

    if (!question) return;

    addMessage(question, "user");
    userInput.value = "";

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question
            })
        });

        if (!response.ok) {
            throw new Error("Server request failed");
        }

        const data = await response.json();
        addMessage(data.answer, "bot");

    } catch (error) {
        addMessage("Error: Could not connect to server.", "bot");
        console.error(error);
    }

    userInput.focus();
});

// ==========================================
// Token vocabulary
// ==========================================

async function loadRandomTokens() {
    randomTokensBtn.disabled = true;

    try {
        const response = await fetch("/api/random-tokens");

        if (!response.ok) {
            throw new Error("Failed to load tokens");
        }

        const data = await response.json();

        tokenGrid.replaceChildren();

        data.tokens.forEach(token => {
            const item = document.createElement("div");

            item.className = "token-item";
            item.textContent = token;

            tokenGrid.appendChild(item);
        });

    } catch (error) {
        console.error(error);
        tokenGrid.textContent = "Failed to load tokens.";
    } finally {
        randomTokensBtn.disabled = false;
    }
}

// ==========================================
// Training examples
// ==========================================

async function loadRandomExamples() {
    randomExamplesBtn.disabled = true;

    try {
        const response = await fetch("/api/random-examples");

        if (!response.ok) {
            throw new Error("Failed to load examples");
        }

        const data = await response.json();

        examplesList.replaceChildren();

        data.examples.forEach(example => {
            const container = document.createElement("div");
            container.className = "example-item";

            const question = document.createElement("div");
            question.className = "example-question";
            question.textContent = "Q: " + example.question;

            const answer = document.createElement("div");
            answer.className = "example-answer";
            answer.textContent = "A: " + example.answer;

            container.appendChild(question);
            container.appendChild(answer);

            examplesList.appendChild(container);
        });

    } catch (error) {
        console.error(error);
        examplesList.textContent = "Failed to load examples.";
    } finally {
        randomExamplesBtn.disabled = false;
    }
}

// ==========================================
// Button events and initialization
// ==========================================

randomTokensBtn.addEventListener("click", loadRandomTokens);
randomExamplesBtn.addEventListener("click", loadRandomExamples);

// Load random data when the page opens
loadRandomTokens();
loadRandomExamples();


 // ==========================================
 // Example questions
 // ==========================================

const examplePrompts = document.getElementById("example-prompts");
const promptList = document.getElementById("prompt-list");
const shufflePromptsBtn = document.getElementById("shuffle-prompts-btn");
const togglePromptsBtn = document.getElementById("toggle-prompts-btn");

const exampleQuestions = [
    "What do you want to do?",
    "What do you like?",
    "Hello!",
    "How is the weather?",
    "Do you like music?",
    "What can you do?",
    "Are you happy?",
    "How are you today?",
    "Can you help me?"
];

function loadExampleQuestions() {
    const selected = [...exampleQuestions]
        .sort(() => Math.random() - 0.5)
        .slice(0, 3);

    promptList.replaceChildren();

    selected.forEach(question => {
        const button = document.createElement("button");

        button.type = "button";
        button.className = "prompt-item";
        button.textContent = question;

        button.addEventListener("click", () => {
            userInput.value = question;
            userInput.focus();
        });

        promptList.appendChild(button);
    });
}

shufflePromptsBtn.addEventListener("click", loadExampleQuestions);

togglePromptsBtn.addEventListener("click", () => {
    const collapsed = examplePrompts.classList.toggle("collapsed");

    togglePromptsBtn.textContent = collapsed ? "▲" : "▼";
    togglePromptsBtn.setAttribute("aria-expanded", String(!collapsed));
    togglePromptsBtn.setAttribute(
        "aria-label",
        collapsed ? "Expand example questions" : "Collapse example questions"
    );
});

loadExampleQuestions();

