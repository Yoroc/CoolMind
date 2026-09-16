import coolmind

def main():
    engine = coolmind.CoolEngine()
    engine.start_monitoring()
    
    print("CoolMind AI Ready. Type 'exit' to quit.")
    while True:
        prompt = input("\nYou: ")
        if prompt.lower() == 'exit':
            break
        response = engine.generate(prompt)
        print(f"AI: {response}")
    
    engine.stop_monitoring()

if __name__ == "__main__":
    main()