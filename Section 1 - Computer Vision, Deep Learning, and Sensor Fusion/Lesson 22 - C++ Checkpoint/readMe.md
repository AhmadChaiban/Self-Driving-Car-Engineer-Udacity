## Challenge 1

### stdout
Let's start with an easy one. In this quiz, I'm giving you a main function. All I want you to do is write no more steering wheels to stdout.

Just as a reminder, anything written to stdout will appear below the code editor after you test or submit your code.

Here's my solution:

```c++
#include <iostream>

int main() {
  std::cout << "no more steering wheels" << std::endl;
  return 0;
}
```

std refers to namespace std and the << operator passes sequences of characters to stdout. In fact, two sequences of characters get passed to stdout: "no more steering wheels" and std::endl, a newline character (std::endl also flushes the buffer).

You may be used to adding a using namespace std line after your include statements - however, under the Google C++ style guide, you should not utilize using-directives to make all names from a namespace available. You could instead use using std::cout; and using std::endl; near the top so that you do not have to re-write std:: before each instance of these - it does not make much of a difference here, but could be more helpful when you use them multiple times!

It's worth noting that "no more steering wheels" is not a string like a Python string, rather it's a char [] - a sequence of characters.

Time for another! Let's control some flow.

## Challenge 2

### Control Flow
In this quiz, I want you to write a `for` loop. Your loop will live inside a function that will be passed an int, n, and a string, str. Your loop should print the str n times.

There's another twist here - you will not need to touch main.cpp this time. Instead, your code will be written inside the eponymous function in PrintString.cpp. Make note of the way that main.cpp #includes PrintString.h. You'll be #includeing files soon.

Here's my solution:

```c++
#include "PrintString.h"

#include <iostream>
#include <string>

void PrintString(std::string str, int n) {
  for (int i = 0; i < n; ++i) {
    std::cout << str << std::endl;
  }
}
```

## Challenge 3

### Functions
In this quiz, I want you to write a function called factorial that returns an int. A factorial is a mathematical operation designated by a !! operator and returns the product of a number and every whole number between it and 0. For example:

    4! = 4 * 3 * 2 * 1 = 24

You'll write your code within Factorial.cpp. Good luck!

Here's my solution:

```c++
#include "Factorial.h"

int Factorial(int n) {
  int result = 1;

  for (int i = n; i > 0; --i) {
    result *= i;
  }

  return result;
}
```

I took advantage of the *= operator to multiply result by a lower int and then reset itself to the new value simultaneously.

Keep going! Only two challenges left!

## Challenge 4

### Class Basics

In this challenge, I want to flip the tables a bit. In the previous quizzes, I've been giving you header files and asking you to implement methods from them. In this quiz, I'm going to give you a .cpp file and ask you to write the corresponding header file.

In Car.cpp, you'll find the implementation of a simple Car class. This is a very unreliable car that has a 50/50 chance of being broken after every drive.

I want you to examine Car.cpp and write the corresponding header file, Car.h. The code you'll find below won't compile without a working header file. Check out the compiler errors and make it work!

Note: In the below quiz, we've added comments over the top of the function definitions in the .cpp file, although typically these comments should be placed over the function declarations in the .h file instead - we've placed them here since .h is not implemented yet.

Here's my solution:

```c++
#ifndef CAR_H
#define CAR_H

class Car {    
 public:
  Car();
  void wearAndTear();
  bool drive();
  void fix();
 private:
  bool in_working_condition_;
};
```

#endif  // CAR_H
The Car class is pretty straightforward. The trickiest part, I found, was making sure that the constructor was defined.

Note, the trailing _ on in_working_condition_ is common tactic for designating private properties in C++.

Ok! One more challenge to go.

## Challenge 5 

### References

Lower level languages, like C++, give you control over the way your objects are stored and accessed in memory. As a comparison, let's start by looking at how Python hides memory access from you.
```python
def dict_modifier(d, key):
  d.pop(key, None)

sample_dict = {'some_key': 'some value'}
dict_modifier(sample_dict, 'some_key')
print(sample_dict)  # {}
```

In this case, dict_modifier removes a key from the dictionary passed to it. In effect, it treats the dictionary argument as a reference.

You may want to draw the conclusion that Python functions always treat arguments as references, but that's not true.

```python
def adder(n):
  n += 1

i = 1
adder(i)
print(i)  # 1
```

Python does not pass primitives as references. As a result, adder receives a copy of i and i remains unchanged outside the function.

To be fair, the way Python handles arguments is perfectly reasonable. But what's happening behind the scenes is not perfectly obvious from the syntax alone.

C++, on the other hand, makes you decide how you'd like to treat function arguments. It also gives you control over how you'd like to access objects in memory, whether that's by simply referring to their address in memory or by their actual value.

### Pass by Reference
In this last challenge, I want you to finish the doubler function that doubles an int passed to it as a reference. You'll need to finish defining the function's parameters and body.

Here's my solution:

`Doubler.h`

```c++
#ifndef DOUBLER_H
#define DOUBLER_H

void Doubler(int& n);

#endif  // DOUBLER_H
```

`Doubler.cpp`

```c++
#include "Doubler.h"

void Doubler(int& n) {
  n *= 2;
}
```

The tricky part here was making sure to use the & operator to pass n as a reference.

