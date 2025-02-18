/*
   Arduino based SCARA Robot GUI
   by Dejan, www.HowToMechatronics.com AccelStepper: http://www.airspayce.com/mikem/arduino/AccelStepper/index.html

*/
import processing.serial.*;
import controlP5.*; 
import static processing.core.PApplet.*;

Serial myPort;
ControlP5 cp5; // controlP5 object

int j1Slider = 0;
int j2Slider = 0;
int j3Slider = 0;
int zSlider = 100;
int j1JogValue = 0;
int j2JogValue = 0;
int j3JogValue = 0;
int zJogValue = 0;
int speedSlider = 500;
int accelerationSlider = 500;
int gripperValue = 180;
int gripperAdd=180;
int positionsCounter = 0;


int saveStatus = 0;
int runStatus = 0;

int slider1Previous = 0;
int slider2Previous = 0;
int slider3Previous = 0;
int sliderzPrevious = 100;
int speedSliderPrevious = 500;
int accelerationSliderPrevious = 500;
int gripperValuePrevious = 100;

boolean activeIK = false;

int xP=365;
int yP=0;
int zP=100;
float L1 = 228; // L1 = 228mm
float L2 = 136.5; // L2 = 136.5mm
float theta1, theta2, phi, z;

String[] positions = new String[100];

String data;

void setup() {

  size(960, 800);
  //myPort = new Serial(this, "COM3", 115200);
  
  cp5 = new ControlP5(this);

  PFont pfont = createFont("Arial", 25, true); // use true/false for smooth/no-smooth
  ControlFont font = new ControlFont(pfont, 22);
  ControlFont font2 = new ControlFont(pfont, 25);

  //J1 controls
  cp5.addSlider("j1Slider")
    .setPosition(110, 190)
    .setSize(270, 30)
    .setRange(-90, 266) // Slider range, corresponds to Joint 1 or theta1 angle that the robot can move to
    .setColorLabel(#3269c2)
    .setFont(font)
    .setCaptionLabel("")
    ;
  cp5.addButton("j1JogMinus")
    .setPosition(110, 238)
    .setSize(90, 40)
    .setFont(font)
    .setCaptionLabel("JOG-")
    ;
  cp5.addButton("j1JogPlus")
    .setPosition(290, 238)
    .setSize(90, 40)
    .setFont(font)
    .setCaptionLabel("JOG+")
    ;
  cp5.addNumberbox("j1JogValue")
    .setPosition(220, 243)
    .setSize(50, 30)
    .setRange(0, 20)
    .setFont(font)
    .setMultiplier(0.1)
    .setValue(1)
    .setDirection(Controller.HORIZONTAL) // change the control direction to left/right
    .setCaptionLabel("")
    ;

  //J2 controls
  cp5.addSlider("j2Slider")
    .setPosition(110, 315)
    .setSize(270, 30)
    .setRange(-150, 150)
    .setColorLabel(#3269c2)
    .setFont(font)
    .setCaptionLabel("")
    ;
  cp5.addButton("j2JogMinus")
    .setPosition(110, 363)
    .setSize(90, 40)
    .setFont(font)
    .setCaptionLabel("JOG-")
    ;
  cp5.addButton("j2JogPlus")
    .setPosition(290, 363)
    .setSize(90, 40)
    .setFont(font)
    .setCaptionLabel("JOG+")
    ;
  cp5.addNumberbox("j2JogValue")
    .setPosition(220, 368)
    .setSize(50, 30)
    .setRange(0, 20)
    .setFont(font)
    .setMultiplier(0.1)
    .setValue(1)
    .setDirection(Controller.HORIZONTAL) // change the control direction to left/right
    .setCaptionLabel("")
    ;
  //J3 controls
  cp5.addSlider("j3Slider")
    .setPosition(110, 440)
    .setSize(270, 30)
    .setRange(-162, 162)
    .setColorLabel(#3269c2)
    .setFont(font)
    .setCaptionLabel("")
    ;
  cp5.addButton("j3JogMinus")
    .setPosition(110, 493)
    .setSize(90, 40)
    .setFont(font)
    .setCaptionLabel("JOG-")
    ;
  cp5.addButton("j3JogPlus")
    .setPosition(290, 493)
    .setSize(90, 40)
    .setFont(font)
    .setCaptionLabel("JOG+")
    ;
  cp5.addNumberbox("j3JogValue")
    .setPosition(220, 493)
    .setSize(50, 30)
    .setRange(0, 20)
    .setFont(font)
    .setMultiplier(0.1)
    .setValue(1)
    .setDirection(Controller.HORIZONTAL) // change the control direction to left/right
    .setCaptionLabel("")
    ;

  //Z controls
  cp5.addSlider("zSlider")
    .setPosition(110, 565)
    .setSize(270, 30)
    .setRange(0, 150)
    .setColorLabel(#3269c2)
    .setFont(font)
    .setCaptionLabel("")
    ;
  cp5.addButton("zJogMinus")
    .setPosition(110, 618)
    .setSize(90, 40)
    .setFont(font)
    .setCaptionLabel("JOG-")
    ;
  cp5.addButton("zJogPlus")
    .setPosition(290, 618)
    .setSize(90, 40)
    .setFont(font)
    .setCaptionLabel("JOG+")
    ;
  cp5.addNumberbox("zJogValue")
    .setPosition(220, 618)
    .setSize(50, 30)
    .setRange(0, 20)
    .setFont(font)
    .setMultiplier(0.1)
    .setValue(1)
    .setDirection(Controller.HORIZONTAL) // change the control direction to left/right
    .setCaptionLabel("")
    ;


  cp5.addTextfield("xTextfield")
    .setPosition(530, 205)
    .setSize(70, 40)
    .setFont(font)
    .setColor(255)
    .setCaptionLabel("")
    ;
  cp5.addTextfield("yTextfield")
    .setPosition(680, 205)
    .setSize(70, 40)
    .setFont(font)
    .setColor(255)
    .setCaptionLabel("")
    ;
  cp5.addTextfield("zTextfield")
    .setPosition(830, 205)
    .setSize(70, 40)
    .setFont(font)
    .setColor(255)
    .setCaptionLabel("")
    ;

  cp5.addButton("move")
    .setPosition(590, 315)
    .setSize(240, 45)
    .setFont(font)
    .setCaptionLabel("MOVE TO POSITION")
    ;

  cp5.addButton("savePosition")
    .setPosition(470, 520)
    .setSize(215, 50)
    .setFont(font2)
    .setCaptionLabel("SAVE POSITION")
    ;

  cp5.addButton("run")
    .setPosition(725, 520)
    .setSize(215, 50)
    .setFont(font2)
    .setCaptionLabel("RUN PROGRAM")
    ;

  cp5.addButton("updateSA")
    .setPosition(760, 590)
    .setSize(150, 40)
    .setFont(font)
    .setCaptionLabel("(Update)")
    ;

  cp5.addButton("clearSteps")
    .setPosition(490, 650)
    .setSize(135, 40)
    .setFont(font)
    .setCaptionLabel("(CLEAR)")
    ;

  //Z controls
  cp5.addSlider("speedSlider")
    .setPosition(490, 740)
    .setSize(180, 30)
    .setRange(500, 4000)
    .setColorLabel(#3269c2)
    .setFont(font)
    .setCaptionLabel("")
    ;

  cp5.addSlider("accelerationSlider")
    .setPosition(720, 740)
    .setSize(180, 30)
    .setRange(500, 4000)
    .setColorLabel(#3269c2)
    .setFont(font)
    .setCaptionLabel("")
    ;
  cp5.addSlider("gripperValue")
    .setPosition(605, 445)
    .setSize(190, 30)
    .setRange(0, 100)
    .setColorLabel(#3269c2)
    .setFont(font)
    .setCaptionLabel("")
    ;
}

void draw() { 
  background(#F2F2F2); // background black
  textSize(26);
  fill(33);
  text("Forward Kinematics", 120, 135); 
  text("Inverse Kinematics", 590, 135); 
  textSize(40);
  text("SCARA Robot Control", 260, 60); 
  textSize(45);
  text("J1", 35, 250); 
  text("J2", 35, 375);
  text("J3", 35, 500);
  text("Z", 35, 625);
  textSize(22);
  text("Speed", 545, 730);
  text("Acceleration", 745, 730);

  //println("PREV: "+accelerationSlider);
  fill(speedSlider);
  fill(accelerationSlider);
  fill(j1Slider);
  fill(j2Slider);
  fill(j3Slider);
  fill(zSlider);
  fill(j1JogValue);
  fill(j2JogValue);
  fill(j3JogValue);
  fill(zJogValue);
  fill(gripperValue);


  updateData();
  //println(data);

  saveStatus=0; // keep savePosition variable 0 or false. See, when button SAVE pressed it makes the value 1, which indicates to store the value in the arduino code

  // If slider moved, calculate new position of X,Y and Z with forward kinematics
  if (slider1Previous != j1Slider) {
    if (activeIK == false) {     // Check whether the inverseKinematics mode is active, Executre Forward kinematics only if inverseKinematics mode is off or false
      theta1 = round(cp5.getController("j1Slider").getValue()); // get the value from the slider1
      theta2 = round(cp5.getController("j2Slider").getValue());
      forwardKinematics();
      myPort.write(data);
    }
  }
  slider1Previous = j1Slider;

  if (slider2Previous != j2Slider) {
    if (activeIK == false) {     // Check whether the inverseKinematics mode is active, Executre Forward kinematics only if inverseKinematics mode is off or false
      theta1 = round(cp5.getController("j1Slider").getValue()); // get the value from the slider1
      theta2 = round(cp5.getController("j2Slider").getValue());
      forwardKinematics();
      myPort.write(data);
    }
  }
  slider2Previous = j2Slider;

  if (slider3Previous != j3Slider) {
    if (activeIK == false) {     // Check whether the inverseKinematics mode is active, Executre Forward kinematics only if inverseKinematics mode is off or false
      theta1 = round(cp5.getController("j1Slider").getValue()); // get the value from the slider1
      theta2 = round(cp5.getController("j2Slider").getValue());
      forwardKinematics();
      myPort.write(data);
    }
  }
  slider3Previous = j3Slider;

  if (sliderzPrevious != zSlider) {
    if (activeIK == false) {     // Check whether the inverseKinematics mode is active, Executre Forward kinematics only if inverseKinematics mode is off or false
      zP = round(cp5.getController("zSlider").getValue());
      myPort.write(data);
    }
  }
  sliderzPrevious = zSlider;

  if (gripperValuePrevious != gripperValue) {
    if (activeIK == false) {     // Check whether the inverseKinematics mode is active, Executre Forward kinematics only if inverseKinematics mode is off or false
      gripperAdd = round(cp5.getController("gripperValue").getValue());
      gripperValue=gripperAdd+50;
      updateData();
      println(data);
      myPort.write(data);
    }
  }
  gripperValuePrevious = gripperValue;
  activeIK = false; // deactivate inverseKinematics so the above if statements can be executed the next interation

  fill(33);
  textSize(32);
  text("X: ", 500, 290);
  text(xP, 533, 290);
  text("Y: ", 650, 290);
  text(yP, 685, 290);
  text("Z: ", 800, 290);
  text(zP, 835, 290);
  textSize(26);
  text("Gripper", 650, 420);
  text("CLOSE", 510, 470);
  text("OPEN", 810, 470);
  textSize(18);

  if (positionsCounter >0 ) {
    text(positions[positionsCounter-1], 460, 630);
    text("Last saved position: No."+(positionsCounter-1), 460, 600);
  } else {
    text("Last saved position:", 460, 600);
    text("None", 460, 630);
  }
}

 // FORWARD KINEMATICS
void forwardKinematics() {
  float theta1F = theta1 * PI / 180;   // degrees to radians
  float theta2F = theta2 * PI / 180;
  xP = round(L1 * cos(theta1F) + L2 * cos(theta1F + theta2F));
  yP = round(L1 * sin(theta1F) + L2 * sin(theta1F + theta2F));
}

 // INVERSE KINEMATICS
void inverseKinematics(float x, float y) {
  theta2 = acos((sq(x) + sq(y) - sq(L1) - sq(L2)) / (2 * L1 * L2));
  if (x < 0 & y < 0) {
    theta2 = (-1) * theta2;
  }
  
  theta1 = atan(x / y) - atan((L2 * sin(theta2)) / (L1 + L2 * cos(theta2)));
  
  theta2 = (-1) * theta2 * 180 / PI;
  theta1 = theta1 * 180 / PI;

 // Angles adjustment depending in which quadrant the final tool coordinate x,y is
  if (x >= 0 & y >= 0) {       // 1st quadrant
    theta1 = 90 - theta1;
  }
  if (x < 0 & y > 0) {       // 2nd quadrant
    theta1 = 90 - theta1;
  }
  if (x < 0 & y < 0) {       // 3d quadrant
    theta1 = 270 - theta1;
    phi = 270 - theta1 - theta2;
    phi = (-1) * phi;
  }
  if (x > 0 & y < 0) {       // 4th quadrant
    theta1 = -90 - theta1;
  }
  if (x < 0 & y == 0) {
    theta1 = 270 + theta1;
  }
  
  // Calculate "phi" angle so gripper is parallel to the X axis
  phi = 90 + theta1 + theta2;
  phi = (-1) * phi;

  // Angle adjustment depending in which quadrant the final tool coordinate x,y is
  if (x < 0 & y < 0) {       // 3d quadrant
    phi = 270 - theta1 - theta2;
  }
  if (abs(phi) > 165) {
    phi = 180 + phi;
  }

  theta1=round(theta1);
  theta2=round(theta2);
  phi=round(phi);
  
  cp5.getController("j1Slider").setValue(theta1);
  cp5.getController("j2Slider").setValue(theta2);
  cp5.getController("j3Slider").setValue(phi);
  cp5.getController("zSlider").setValue(zP);
}

void controlEvent(ControlEvent theEvent) {  

  if (theEvent.isController()) { 
    println(theEvent.getController().getName());
  }
}

public void xTextfield(String theText) {
  //If we enter a value into the Textfield, read the value, convert to integer, set the inverseKinematics mode active
  xP=Integer.parseInt(theText);
  activeIK = true;
  inverseKinematics(xP, yP); // Use inverse kinematics to calculate the J1(theta1), J2(theta2), and J3(phi) positions
  //activeIK = false;
  println("Test; theta1: "+theta1+" theta2: "+theta2);
}
public void yTextfield(String theText) {
  yP=Integer.parseInt(theText);
  activeIK = true;
  inverseKinematics(xP, yP);
  //activeIK = false;
}
public void zTextfield(String theText) {
  zP=Integer.parseInt(theText);
  activeIK = true;
  inverseKinematics(xP, yP);
}

public void j1JogMinus() {
  int a = round(cp5.getController("j1Slider").getValue());
  a=a-j1JogValue;
  cp5.getController("j1Slider").setValue(a);
}
//J1 control
public void j1JogPlus() {
  int a = round(cp5.getController("j1Slider").getValue());
  a=a+j1JogValue;
  cp5.getController("j1Slider").setValue(a);
}
//J2 control
public void j2JogMinus() {
  int a = round(cp5.getController("j2Slider").getValue());
  a=a-j2JogValue;
  cp5.getController("j2Slider").setValue(a);
}
public void j2JogPlus() {
  int a = round(cp5.getController("j2Slider").getValue());
  a=a+j2JogValue;
  cp5.getController("j2Slider").setValue(a);
}
//J3 control
public void j3JogMinus() {
  int a = round(cp5.getController("j3Slider").getValue());
  a=a-j3JogValue;
  cp5.getController("j3Slider").setValue(a);
}
public void j3JogPlus() {
  int a = round(cp5.getController("j3Slider").getValue());
  a=a+j3JogValue;
  cp5.getController("j3Slider").setValue(a);
}
//J3 control
public void zJogMinus() {
  int a = round(cp5.getController("zSlider").getValue());
  a=a-zJogValue;
  cp5.getController("zSlider").setValue(a);
}
public void zJogPlus() {
  int a = round(cp5.getController("zSlider").getValue());
  a=a+zJogValue;
  ;
  cp5.getController("zSlider").setValue(a);
}

public void move() {

  myPort.write(data);
  println(data);
}

public void savePosition() {
  // Save the J1, J2, J3 and Z position in the array 
  positions[positionsCounter]="J1="+str(round(cp5.getController("j1Slider").getValue()))
    +"; J2=" + str(round(cp5.getController("j2Slider").getValue()))
    +"; J3="+str(round(cp5.getController("j3Slider").getValue()))
    +"; Z="+str(round(cp5.getController("zSlider").getValue()));
  positionsCounter++;
  saveStatus = 1;
  updateData();
  myPort.write(data);
  saveStatus=0;
}

public void run() {

  if (runStatus == 0) {
    cp5.getController("run").setCaptionLabel("STOP");
    cp5.getController("run").setColorLabel(#e74c3c);

    runStatus = 1;
  } else if (runStatus == 1) {
    runStatus = 0;
    cp5.getController("run").setCaptionLabel("RUN PROGRAM");
    cp5.getController("run").setColorLabel(255);
  }
  updateData();
  myPort.write(data);
}
public void updateSA() {
  myPort.write(data);
}
public void clearSteps() {
  saveStatus = 2; // clear all steps / program
  updateData();
  myPort.write(data);
  println("Clear: "+data);
  positionsCounter=0;
  saveStatus = 0;
}

public void updateData() {
  data = str(saveStatus)
    +","+str(runStatus)
    +","+str(round(cp5.getController("j1Slider").getValue())) 
    +","+str(round(cp5.getController("j2Slider").getValue()))
    +","+str(round(cp5.getController("j3Slider").getValue()))
    +","+str(round(cp5.getController("zSlider").getValue()))
    +","+str(gripperValue)
    +","+str(speedSlider)
    +","+str(accelerationSlider);
}
