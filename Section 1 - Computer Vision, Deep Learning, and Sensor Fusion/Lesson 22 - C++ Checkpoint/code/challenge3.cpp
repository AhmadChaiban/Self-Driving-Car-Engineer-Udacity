#include "Factorial.h"

// Your code goes here! 
// See Factorial.h for your method's signature (eg, its args and return value).

int Factorial(int n){
    int prod = 1;
    for(int i=1; i<=n; i++){
        prod = prod * i;
    }
    return prod;
}
