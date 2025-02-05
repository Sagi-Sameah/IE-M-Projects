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

    print("\nWould you like to learn about nested 'if' statements? (yes/no)")
    if input().lower() == 'yes':
        print("\nNested 'if' statements allow you to have another 'if' statement inside an existing one.")
        print("\nExample:")
        print("if temperature > 30:")
        print("    if temperature > 40:")
        print("        print('It\'s extremely hot outside!')")
        print("    else:")
        print("        print('It\'s hot outside!')")
        print("\nYou can try this in your own way!")

        print("\nWould you like to try a nested 'if' statement? (yes/no)")
        if input().lower() == 'yes':
            print("\nEnter a temperature for the nested 'if' example:")
            temp = int(input())
            if temp > 30:
                if temp > 40:
                    print("It's extremely hot outside!")
                else:
                    print("It's hot outside!")
            else:
                print("It's not that hot outside.")


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

    print("\nWould you like to learn about nested 'for' loops? (yes/no)")
    if input().lower() == 'yes':
        print(
            "\nNested 'for' loops allow you to have one 'for' loop inside another. This is useful for tasks like handling grids.")
        print("\nExample:")
        print("for i in range(3):")
        print("    for j in range(2):")
        print("        print(f'Row {i}, Column {j}')")
        print("\nThis will print the positions in a 3x2 grid!")

        print("\nWould you like to try a nested 'for' loop? (yes/no)")
        if input().lower() == 'yes':
            print("\nEnter the number of rows for the nested 'for' loop:")
            rows = int(input())
            print("Enter the number of columns for the nested 'for' loop:")
            cols = int(input())
            for i in range(rows):
                for j in range(cols):
                    print(f"Row {i}, Column {j}")


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

    print("\nWould you like to learn about nested 'while' loops? (yes/no)")
    if input().lower() == 'yes':
        print("\nNested 'while' loops allow you to have one 'while' loop inside another.")
        print("\nExample:")
        print("i = 0")
        print("while i < 2:")
        print("    j = 0")
        print("    while j < 2:")
        print("        print(f'Outer loop: {i}, Inner loop: {j}')")
        print("        j += 1")
        print("    i += 1")
        print("\nThis will print a 2x2 grid of outer and inner loop values!")

        print("\nWould you like to try a nested 'while' loop? (yes/no)")
        if input().lower() == 'yes':
            print("\nEnter the number of iterations for the outer 'while' loop:")
            outer = int(input())
            print("Enter the number of iterations for the inner 'while' loop:")
            inner = int(input())
            i = 0
            while i < outer:
                j = 0
                while j < inner:
                    print(f"Outer loop: {i}, Inner loop: {j}")
                    j += 1
                i += 1

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

    print("\nWould you like to learn about nested 'if' statements? (yes/no)")
    if input().lower() == 'yes':
        print("\nNested 'if' statements can be used in a switch-like structure.")
        print("\nExample:")
        print("day = 'Monday'")
        print("if day == 'Monday':")
        print("    if day == 'Monday' and time == 'Morning':")
        print("        print('Good Morning! Start of the week')")
        print("    else:")
        print("        print('Start of the week')")
        print("\nThis shows how to combine multiple conditions!")

        print("\nWould you like to try a nested 'if' statement in a switch? (yes/no)")
        if input().lower() == 'yes':
            print("\nEnter a day of the week:")
            day = input().capitalize()
            print("Enter the time of day (Morning/Afternoon/Evening):")
            time = input().capitalize()
            if day == 'Monday':
                if time == 'Morning':
                    print("Good Morning! Start of the week")
                else:
                    print("Start of the week")
            else:
                print("Middle of the week")


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
            print("Thanks for learning with us! Happy coding!")
            break


# Start the interactive guide
main()

