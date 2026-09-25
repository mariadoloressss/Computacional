#include <iostream>
#include <math.h>
#include "complex.h" //include library to work with complex numbers. 
                     //REMEMBER: the file complex.h needs to be in the same directory as the current file!

using namespace std;

int main()
{
    double x=3.;

    //declare complex numbers
    fcomplex a,b,c;

    //One way of defining the complex number is by assigning a value to its members (r for real and i for imaginary part)
    b.r=0.5;
    b.i=0.5;

    //Another way of defining the complex number is through the function Complex(real part, imaginary part)
    a=Complex(1.0,3.0);

    //complex number multiplication
    c=Cmul(a,b);

    //print the multiplication operation
    cout << "( " << a.r << " + " << a.i << "i ) * ( " << b.r << " + " << b.i << " i ) = " << c.r << " + " << c.i << "i" << endl;

    //complex number addition
    c=Cadd(a,b);

    //print the addition operation
    cout << "( " << a.r << " + " << a.i << " i ) + ( " << b.r << " + " << b.i << "i ) = " << c.r << " + " << c.i << "i" << endl;

    //Multiplication of a real number by a complex number
    c=RCmul(x,b);

    //print the multiplication operation
    cout << x << " * (" << b.r << " + " << b.i << "i ) = " << c.r << " + " << c.i << "i" << endl;

    //abs
    x=Cabs(a);
    cout << "abs( " << a.r << " + " << a.i << "i ) = " << x << endl;

    //a*a^+
    c=Cmul(a,Conjg(a));
    cout << "( " << a.r << " + " << a.i << "i ) * ( " << a.r << " - " << a.i << "i ) = " << c.r << " + " << c.i << "i" << endl;

}
