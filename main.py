import os
from dotenv import load_dotenv
from print_color import print

from src.assistant import DocumentAssistant


def print_header():
    """Print the application header."""
    print("\n" + "=" * 60)
    print("EvidenceFlow AI Document Assistant", color="blue")
    print("=" * 60 + "\n")


def print_help():
    """Print commands and example questions."""
    print("\nAVAILABLE COMMANDS:", color="blue")
    print("  /help: Show this help message")
    print("  /docs: List available documents")
    print("  /quit: Exit the assistant")
    print("\nExample queries:")
    print("  1. What is the total amount in invoice INV-001?")
    print("  2. Summarize all contracts")
    print("  3. Calculate the sum of all invoice totals")
    print("  4. Find documents with amounts over $50,000")
    print()


def list_documents(assistant: DocumentAssistant):
    """List the available sample documents."""
    print("\nAVAILABLE DOCUMENTS:", color="blue")
    print("-" * 40)

    for doc_id, doc in assistant.retriever.documents.items():
        print(f"ID: {doc_id}")
        print(f"Title: {doc.title}")
        print(f"Type: {doc.doc_type}")
        if "total" in doc.metadata:
            print(f"Total: ${doc.metadata['total']:,.2f}")
        elif "amount" in doc.metadata:
            print(f"Amount: ${doc.metadata['amount']:,.2f}")
        elif "value" in doc.metadata:
            print(f"Value: ${doc.metadata['value']:,.2f}")
        print("-" * 40)


def main():
    """Run the interactive document assistant."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        return

    print_header()

    print(" INITIALIZING ASSISTANT...", color="green")
    assistant = DocumentAssistant(
        openai_api_key=api_key,
        model_name=os.getenv("MODEL_NAME", "gpt-4o"),
        temperature=float(os.getenv("TEMPERATURE", "0.1")),
        reasoning_effort=os.getenv("REASONING_EFFORT") or None,
        session_storage_path=os.getenv("SESSION_STORAGE_PATH", "./sessions"),
        logs_dir=os.getenv("LOGS_DIR", "./logs"),
    )

    user_id = input("Enter your user ID (or press Enter for 'demo_user'): ").strip() or "demo_user"
    session_id = assistant.start_session(user_id)
    print(f"Session started: {session_id}")

    print_help()

    while True:
        try:
            user_input = input("\nEnter Message: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("\nGoodbye!", color="blue")
                break
            elif user_input.lower() == "/help":
                print_help()
                continue
            elif user_input.lower() == "/docs":
                list_documents(assistant)
                continue

            print("\nProcessing...", color="yellow")
            result = assistant.process_message(user_input)

            if result["success"]:
                print("\nAssistant:", end=" ")

                if result.get("response"):
                    print(result["response"])
                if result.get("intent"):
                    intent = result["intent"]
                    print(f"\nINTENT: {intent['intent_type']}", color="green")
                if result.get("active_documents"):
                    print(f"\nSOURCES: {', '.join(result['active_documents'])}", color="blue")
                if result.get("tools_used"):
                    print(f"\nTOOLS USED: {', '.join(result['tools_used'])}", color="magenta")
                if result.get("summary"):
                    print(f"\nCONVERSATION SUMMARY: {result['summary']}", color="cyan")

            else:
                print(f"\nError: {result.get('error', 'Unknown error')}", color="red")

        except KeyboardInterrupt:
            print("\n\nGoodbye!", color="blue")
            break
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}", color="red")


if __name__ == "__main__":
    main()
