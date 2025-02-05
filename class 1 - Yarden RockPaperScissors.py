import random

user_choice = 0

while user_choice != -1:
    rounds = int(input("Enter number of rounds"))
    choices = ["rock", "paper", "scissors"]
    user_choice = int(input("Enter 0 for rock, 1 for paper, or 2 for scissors: \nIf you want to exit press -1"))

    # Computer randomly selects a choice
    computer_choice = random.randint(0, 2)

    # Display the choices
    print(f"user: {choices[user_choice]}\ncomputer: {choices[computer_choice]}")

    # Determine the outcome
    if computer_choice == user_choice:
        print("It's a Tie!")
    else:
        if user_choice == 0:  # user chose rock
            if computer_choice == 1:  # computer chose paper
                print("computer wins")
            else:  # computer chose scissors
                print("user wins")

        elif user_choice == 1:  # user chose paper
            if computer_choice == 2:  # computer chose scissors
                print("computer wins")
            else:  # computer chose rock
                print("user wins")

        elif user_choice == 2:  # user chose scissors
            if computer_choice == 0:  # computer chose rock
                print("computer wins")
            else:  # computer chose paper
                print("user wins")

