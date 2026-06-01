from doctor_agent import generate_response

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break

    response = generate_response(user_input)
    print("\nDoctor:\n", response)