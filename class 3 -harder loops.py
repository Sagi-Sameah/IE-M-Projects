def explain_if_statement():
    print("\nThe 'if' statement is used to make decisions in your code.")
    print("It checks if a condition is true or false, and based on that, executes a block of code.")
    print("\nExample:")
    print("if temperature > 30:")
    print("    print('It\'s hot outside!')")
    print("\nIn this example, if the 'temperature' is greater than 30, it will print 'It's hot outside!'")
    print("\nLet's try an 'if' statement. Enter a number for the temperature:")
    temperature = int(input())
    if temperature > 30:
        print("It's hot outside!")
    else:
        print("It's not that hot outside.")

    print("\nAwesome! You're learning well! Let's make it more interesting!")
    print("\nNow, let's combine 'if' with a 'for' loop to create a pattern.")
    print("\nWould you like to learn how to create a star pattern using 'if' + 'for' loop? (yes/no)")
    if input().lower() == 'yes':
        print("\nLet's create a triangle pattern with stars based on the number you input.")
        print("\nExample:")
        print("n = 5")
        print("for i in range(n):")
        print("    if i % 2 == 0:  # For every even row, print stars")
        print("        print('*' * (i + 1))")

        print("\nWould you like to try creating this pattern? (yes/no)")
        if input().lower() == 'yes':
            print("\nEnter a number to define the height of the triangle:")
            n = int(input())
            for i in range(n):
                if i % 2 == 0:
                    print('*' * (i + 1))
        else:
            print("\nGreat! You're already making progress! Keep practicing!")


def explain_for_statement():
    print("\nThe 'for' statement is used to repeat a block of code a certain number of times.")
    print("It is helpful when you know how many times you want to repeat a task.")
    print("\nExample:")
    print("for i in range(5):")
    print("    print(i)")
    print("\nThis will print the numbers from 0 to 4. Now, let's try a 'for' loop.")
    print("Enter a number to repeat a message that many times:")
    count = int(input())
    for i in range(count):
        print(f"This is message number {i + 1}")

    print("\nGreat! You're grasping this concept. Let's take it further!")
    print("\nWould you like to learn about creating more complex patterns using 'for' loops? (yes/no)")
    if input().lower() == 'yes':
        print("\nLet's try to create a square pattern using nested 'for' loops.")
        print("\nExample:")
        print("n = 5")
        print("for i in range(n):")
        print("    for j in range(n):")
        print("        print('*', end=' ')")
        print("    print()")

        print("\nThis will create a square grid of stars. Would you like to try creating a square pattern? (yes/no)")
        if input().lower() == 'yes':
            print("\nEnter a number for the size of the square grid:")
            n = int(input())
            for i in range(n):
                for j in range(n):
                    print('*', end=' ')
                print()
        else:
            print("\nYou're doing awesome! Try these challenges later!")


def explain_while_statement():
    print("\nThe 'while' statement is used to repeat a block of code as long as a condition is true.")
    print("It will keep repeating until the condition becomes false.")
    print("\nExample:")
    print("count = 0")
    print("while count < 3:")
    print("    print(count)")
    print("    count += 1")
    print("\nIn this example, it will print 0, 1, and 2. Let's try a 'while' loop.")
    print("Enter a number to count down from:")
    count = int(input())
    while count > 0:
        print(count)
        count -= 1
    print("Done!")

    print("\nFantastic! You're progressing well. Now, let's combine 'while' and 'if' for something more creative.")
    print("\nWould you like to create a pyramid pattern using 'while' + 'if'? (yes/no)")
    if input().lower() == 'yes':
        print("\nLet's create a pyramid pattern of stars.")
        print("\nExample:")
        print("n = 5")
        print("i = 0")
        print("while i < n:")
        print("    print(' ' * (n - i - 1) + '*' * (2 * i + 1))")
        print("    i += 1")

        print("\nWould you like to try creating a pyramid? (yes/no)")
        if input().lower() == 'yes':
            print("\nEnter a number for the height of the pyramid:")
            n = int(input())
            i = 0
            while i < n:
                print(' ' * (n - i - 1) + '*' * (2 * i + 1))
                i += 1
        else:
            print("\nYou're doing great! Keep experimenting on your own!")


def explain_switch_statement():
    print("\nPython doesn't have a built-in 'switch' statement like other languages.")
    print("But we can use if-elif-else statements to achieve the same result.")
    print("\nExample:")
    print("day = 'Monday'")
    print("if day == 'Monday':")
    print("    print('Start of the week')")
    print("elif day == 'Friday':")
    print("    print('End of the week')")
    print("else:")
    print("    print('Middle of the week')")
    print("\nThis simulates a switch-like structure in Python. Let's try it out!")
    print("Enter a day of the week:")
    day = input().capitalize()  # Capitalize to handle different input formats
    if day == 'Monday':
        print('Start of the week')
    elif day == 'Friday':
        print('End of the week')
    else:
        print('Middle of the week')

    print("\nYou're doing great! Now, let's combine the concepts of 'switch' with loops to make a creative pattern.")
    print("\nWould you like to create a pattern with multiple symbols using 'if' and 'switch'? (yes/no)")
    if input().lower() == 'yes':
        print("\nLet's create a checkerboard pattern with different symbols.")
        print("\nExample:")
        print("n = 5")
        print("for i in range(n):")
        print("    if i % 2 == 0:")
        print("        print('X ' * n)")
        print("    else:")
        print("        print('O ' * n)")

        print("\nWould you like to try creating a checkerboard pattern? (yes/no)")
        if input().lower() == 'yes':
            print("\nEnter the size of the checkerboard:")
            n = int(input())
            for i in range(n):
                if i % 2 == 0:
                    print('X ' * n)
                else:
                    print('O ' * n)
        else:
            print("\nAmazing job! You are mastering these concepts quickly!")


def main():
    print("Welcome to the Python coding guide!")
    while True:
        print("\nWhich topic would you like to learn about?")
        print("1. 'if' statement")
        print("2. 'for' loop")
        print("3. 'while' loop")
        print("4. 'switch' statement")
        choice = input("Enter 1, 2, 3, or 4 to choose a topic: ")

        if choice == '1':
            explain_if_statement()
        elif choice == '2':
            explain_for_statement()
        elif choice == '3':
            explain_while_statement()
        elif choice == '4':
            explain_switch_statement()
        else:
            print("Invalid choice! Please enter a valid number (1-4).")

        print("\nWould you like to learn another topic or continue with the current one? (yes/no): ")
        if input().lower() != 'yes':
            print("Thanks for learning with us! Keep up the great work, and happy coding!")
            break


# Start the interactive guide
main()
