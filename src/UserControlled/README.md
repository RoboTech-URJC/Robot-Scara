# User Controlled Version:
This version uses HowToMechatronics code,to simply:
* Open an user interface for controlling the SCARA,angles,positions...
* Connects arduino serial port to the user interface program for it to receive orders.
## How to run:
### Step 1 : Run arduino program

In [SCARA-ROBOT](https://github.com/RoboTech-URJC/Robot-Scara/tree/main/src/SCARA_Robot) you will find SCARA_ROBOT.ino
,you must run it in **Arduino IDE** after installing AccelStepper library.

### Step 2 : Run user interface
Verify if you have installed **Processing IDE**  [DOWNLOAD](https://processing.org).
In [GUI-for-SCARA](https://github.com/RoboTech-URJC/Robot-Scara/tree/main/src/GUI_for_SCARA_Robot)  you will find GUI_for_SCARA_ROBOT.pde
Open this file in the processing IDE and install every library needed,in this case ControlP5 in library section.Once installed:

**Configure the COM  port** in this file you will find:
```bash
void setup() {
  size(960, 800);
  //myPort = new Serial(this, "COM3", 115200);  // Esta es la línea que necesitas modificar
```
Make sure the program will conect to the correct port (the one with the arduino).

Finally run the program and this will be shown:
![Screenshot from 2025-02-18 11-49-27](https://github.com/user-attachments/assets/e5ae9e98-2779-42b7-ba11-79307d791035)

Now you can succesfully control your scara robot!
