#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <HX711.h>

// ============================================================
// SMART-FOLLOWER SHOPPING CART
// ARDUINO CONTROLLER
// ============================================================
//
// Functions:
//   - Ultrasonic distance measurement
//   - HX711 load-cell measurement
//   - Pi command/watchdog monitoring
//   - Forward/reverse motor control
//   - Motor burst/pulse control
//   - LCD status display
//   - LCD I2C recovery
//
// ============================================================


// ============================================================
// PIN DEFINITIONS
// ============================================================

// Ultrasonic sensor
#define TRIG_PIN 2
#define ECHO_PIN 3

// Motor PWM
#define RPWM 5
#define LPWM 6

// H-bridge enable pins
#define L_EN 7
#define R_EN 8

// Locally constructed H-bridge direction pins
#define L_DIR 9
#define R_DIR 10

// HX711 load cell
#define HX711_DOUT A0
#define HX711_CLK  A1


// ============================================================
// LCD
// ============================================================

#define LCD_ADDRESS 0x27

LiquidCrystal_I2C lcd(LCD_ADDRESS, 16, 2);


// ============================================================
// HX711
// ============================================================

HX711 scale;

// Calibration factor obtained from 1.5 kg calibration test
float calibration_factor = 216262.67;


// ============================================================
// DISTANCE CONTROL
// ============================================================

const float MIN_FOLLOW_DISTANCE_CM = 55.0;
const float MAX_FOLLOW_DISTANCE_CM = 90.0;

const float MAX_DETECTION_DISTANCE_CM = 250.0;


// ============================================================
// DISTANCE SENSOR SETTINGS
// ============================================================

const unsigned long DISTANCE_INTERVAL = 80;
const unsigned long ECHO_TIMEOUT_US = 20000;

const int DISTANCE_BUFFER_SIZE = 5;

// Number of consecutive invalid readings before declaring
// the distance measurement invalid.
const int MAX_INVALID_READINGS = 2;


// ============================================================
// MOTOR SETTINGS
// ============================================================

// Current motor PWM
const int MOTOR_SPEED = 140;

// Motor burst/pulse timing
const unsigned long MOTOR_ON_TIME = 30;
const unsigned long MOTOR_OFF_TIME = 100;

// Safety delay when changing direction
const unsigned long REVERSAL_STOP_TIME = 300;


// ============================================================
// PI WATCHDOG
// ============================================================

const unsigned long PI_TIMEOUT = 3000;


// ============================================================
// PERIODIC TASK INTERVALS
// ============================================================

const unsigned long WEIGHT_INTERVAL = 250;
const unsigned long LCD_INTERVAL = 250;
const unsigned long HEARTBEAT_INTERVAL = 1000;


// ============================================================
// LCD RECOVERY SETTINGS
// ============================================================

// Check whether the LCD I2C backpack is responding.
const unsigned long LCD_I2C_CHECK_INTERVAL = 2000;

// Periodically reinitialize the LCD even when the I2C backpack
// still acknowledges. This handles cases where the backpack
// remains powered but the HD44780 display state becomes corrupted.
const unsigned long LCD_REINIT_INTERVAL = 60000;


// ============================================================
// GLOBAL VARIABLES
// ============================================================

// --------------------------
// Distance
// --------------------------

float distance = 999.0;
float filteredDistance = 999.0;

float distanceBuffer[DISTANCE_BUFFER_SIZE];

int distanceBufferIndex = 0;
int validDistanceSamples = 0;

bool distanceValid = false;

int invalidDistanceReadings = 0;

unsigned long lastDistanceTime = 0;


// --------------------------
// Weight
// --------------------------

float weight = 0.0;

unsigned long lastWeightTime = 0;


// --------------------------
// Pi communication
// --------------------------

bool piCommand = false;

unsigned long lastPiCommandTime = 0;


// --------------------------
// Motor direction
// --------------------------
//
//  1 = Forward
//  0 = Stop
// -1 = Reverse
//

int desiredDirection = 0;
int currentDirection = 0;

bool reversingSafetyDelay = false;

unsigned long reversalStartTime = 0;


// --------------------------
// Motor pulse control
// --------------------------

bool motorPulseState = false;

unsigned long lastMotorPulseTime = 0;


// --------------------------
// LCD
// --------------------------

unsigned long lastLCDUpdate = 0;
unsigned long lastLCDI2CCheck = 0;
unsigned long lastLCDReinit = 0;

bool lcdAvailable = true;

bool lcdRecoveryInProgress = false;


// Store previous LCD contents so unnecessary writes are avoided.
char previousLine1[17] = "";
char previousLine2[17] = "";


// --------------------------
// Heartbeat
// --------------------------

unsigned long lastHeartbeat = 0;


// ============================================================
// LCD I2C CONNECTION TEST
// ============================================================

bool lcdI2CConnected()
{
    Wire.beginTransmission(LCD_ADDRESS);

    byte error = Wire.endTransmission();

    return (error == 0);
}


// ============================================================
// LCD RECOVERY
// ============================================================

void recoverLCD()
{
    if (lcdRecoveryInProgress)
        return;

    lcdRecoveryInProgress = true;

    // Reinitialize the I2C LCD.
    lcd.init();

    delay(5);

    lcd.backlight();

    delay(5);

    lcd.clear();

    delay(5);

    // Force the next normal LCD update to rewrite both lines.
    previousLine1[0] = '\0';
    previousLine2[0] = '\0';

    lcdAvailable = true;

    lastLCDReinit = millis();

    lcdRecoveryInProgress = false;
}


// ============================================================
// INITIAL LCD SETUP
// ============================================================

void initializeLCD()
{
    Wire.begin();

    delay(50);

    lcd.init();

    delay(10);

    lcd.backlight();

    delay(10);

    lcd.clear();

    delay(10);

    lcd.setCursor(0, 0);
    lcd.print("Smart Cart");

    lcd.setCursor(0, 1);
    lcd.print("Starting...");

    delay(1000);

    lcd.clear();

    previousLine1[0] = '\0';
    previousLine2[0] = '\0';

    lcdAvailable = true;

    lastLCDI2CCheck = millis();
    lastLCDReinit = millis();
}


// ============================================================
// WRITE ONE LCD LINE
// ============================================================

void writeLCDLine(byte row, const char *text)
{
    if (!lcdAvailable)
        return;

    char line[17];

    // Copy maximum 16 characters.
    strncpy(line, text, 16);

    // Always terminate.
    line[16] = '\0';

    // Pad unused characters with spaces.
    int length = strlen(line);

    while (length < 16)
    {
        line[length] = ' ';
        length++;
    }

    line[16] = '\0';

    lcd.setCursor(0, row);
    lcd.print(line);
}


// ============================================================
// LCD UPDATE
// ============================================================

void updateLCD()
{
    unsigned long now = millis();

    if (now - lastLCDUpdate < LCD_INTERVAL)
        return;

    lastLCDUpdate = now;


    // --------------------------------------------------------
    // Check I2C connection periodically
    // --------------------------------------------------------

    if (now - lastLCDI2CCheck >= LCD_I2C_CHECK_INTERVAL)
    {
        lastLCDI2CCheck = now;

        if (!lcdI2CConnected())
        {
            lcdAvailable = false;
            recoverLCD();
        }
        else
        {
            lcdAvailable = true;
        }
    }


    // --------------------------------------------------------
    // Periodic LCD reinitialization
    // --------------------------------------------------------

    if (now - lastLCDReinit >= LCD_REINIT_INTERVAL)
    {
        recoverLCD();
    }


    if (!lcdAvailable)
        return;


    // --------------------------------------------------------
    // Convert weight and distance without snprintf("%f")
    // --------------------------------------------------------

    char weightText[8];
    char distanceText[8];

    dtostrf(weight, 4, 2, weightText);

    if (distanceValid)
    {
        dtostrf(filteredDistance, 3, 0, distanceText);
    }
    else
    {
        strcpy(distanceText, "---");
    }


    // --------------------------------------------------------
    // Construct line 1
    // --------------------------------------------------------

    char line1[17];

    snprintf(
        line1,
        sizeof(line1),
        "WT:%s D:%scm",
        weightText,
        distanceText
    );


    // --------------------------------------------------------
    // Construct line 2
    // --------------------------------------------------------

    char line2[17];

    if (!piCommand && (now - lastPiCommandTime >= PI_TIMEOUT))
    {
        snprintf(line2, sizeof(line2), "Pi Timeout");
    }
    else if (!piCommand)
    {
        snprintf(line2, sizeof(line2), "Waiting for Pi");
    }
    else if (!distanceValid)
    {
        snprintf(line2, sizeof(line2), "Out of range");
    }
    else if (reversingSafetyDelay)
    {
        snprintf(line2, sizeof(line2), "Changing Dir");
    }
    else if (desiredDirection == 1)
    {
        snprintf(line2, sizeof(line2), "Forward");
    }
    else if (desiredDirection == -1)
    {
        snprintf(line2, sizeof(line2), "Reverse");
    }
    else
    {
        snprintf(line2, sizeof(line2), "Hold");
    }


    // --------------------------------------------------------
    // Write only when contents change
    // --------------------------------------------------------

    if (strcmp(line1, previousLine1) != 0)
    {
        writeLCDLine(0, line1);

        strncpy(previousLine1, line1, 16);
        previousLine1[16] = '\0';
    }


    if (strcmp(line2, previousLine2) != 0)
    {
        writeLCDLine(1, line2);

        strncpy(previousLine2, line2, 16);
        previousLine2[16] = '\0';
    }
}


// ============================================================
// PI COMMAND UPDATE
// ============================================================

void updatePiCommand()
{
    while (Serial.available() > 0)
    {
        char command = Serial.read();

        if (command == '1')
        {
            piCommand = true;
            lastPiCommandTime = millis();
        }
        else if (command == '0')
        {
            piCommand = false;
            lastPiCommandTime = millis();
        }
    }


    // Watchdog timeout
    if (millis() - lastPiCommandTime > PI_TIMEOUT)
    {
        piCommand = false;
    }
}


// ============================================================
// DISTANCE SAMPLE FILTER
// ============================================================

void addDistanceSample(float newDistance)
{
    distanceBuffer[distanceBufferIndex] = newDistance;

    distanceBufferIndex++;

    if (distanceBufferIndex >= DISTANCE_BUFFER_SIZE)
        distanceBufferIndex = 0;


    if (validDistanceSamples < DISTANCE_BUFFER_SIZE)
        validDistanceSamples++;


    float total = 0.0;

    for (int i = 0; i < validDistanceSamples; i++)
    {
        total += distanceBuffer[i];
    }


    filteredDistance = total / validDistanceSamples;
}


// ============================================================
// DISTANCE SENSOR UPDATE
// ============================================================

void updateDistance()
{
    unsigned long now = millis();

    if (now - lastDistanceTime < DISTANCE_INTERVAL)
        return;

    lastDistanceTime = now;


    // --------------------------------------------------------
    // Trigger ultrasonic sensor
    // --------------------------------------------------------

    digitalWrite(TRIG_PIN, LOW);

    delayMicroseconds(2);

    digitalWrite(TRIG_PIN, HIGH);

    delayMicroseconds(10);

    digitalWrite(TRIG_PIN, LOW);


    // --------------------------------------------------------
    // Read echo
    // --------------------------------------------------------

    unsigned long duration =
        pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);


    // --------------------------------------------------------
    // No echo
    // --------------------------------------------------------

    if (duration == 0)
    {
        distance = 999.0;

        invalidDistanceReadings++;

        // Do NOT destroy filteredDistance here.
        // Preserve the last valid filtered value.

        if (invalidDistanceReadings >= MAX_INVALID_READINGS)
        {
            distanceValid = false;
        }

        return;
    }


    // --------------------------------------------------------
    // Calculate distance
    // --------------------------------------------------------

    distance =
        (duration * 0.0343) / 2.0;


    // --------------------------------------------------------
    // Reject out-of-range measurement
    // --------------------------------------------------------

    if (distance > MAX_DETECTION_DISTANCE_CM ||
        distance < 2.0)
    {
        distance = 999.0;

        invalidDistanceReadings++;

        if (invalidDistanceReadings >= MAX_INVALID_READINGS)
        {
            distanceValid = false;
        }

        return;
    }


    // --------------------------------------------------------
    // Valid reading
    // --------------------------------------------------------

    invalidDistanceReadings = 0;

    distanceValid = true;

    addDistanceSample(distance);
}


// ============================================================
// WEIGHT UPDATE
// ============================================================

void updateWeight()
{
    unsigned long now = millis();

    if (now - lastWeightTime < WEIGHT_INTERVAL)
        return;

    lastWeightTime = now;


    if (scale.wait_ready_timeout(5))
    {
        weight = scale.get_units(1);

        weight = abs(weight);


        // Remove very small fluctuations around zero.
        if (weight < 0.01)
        {
            weight = 0.0;
        }
    }
}


// ============================================================
// DETERMINE DESIRED MOTOR DIRECTION
// ============================================================

void determineDirection()
{
    // --------------------------------------------------------
    // No Pi command
    // --------------------------------------------------------

    if (!piCommand)
    {
        desiredDirection = 0;
        return;
    }


    // --------------------------------------------------------
    // No valid distance
    // --------------------------------------------------------

    if (!distanceValid)
    {
        desiredDirection = 0;
        return;
    }


    // --------------------------------------------------------
    // Too far from user
    // --------------------------------------------------------

    if (filteredDistance > MAX_FOLLOW_DISTANCE_CM)
    {
        desiredDirection = 1;
        return;
    }


    // --------------------------------------------------------
    // Too close to user
    // --------------------------------------------------------

    if (filteredDistance < MIN_FOLLOW_DISTANCE_CM)
    {
        desiredDirection = -1;
        return;
    }


    // --------------------------------------------------------
    // Inside following zone
    // --------------------------------------------------------

    desiredDirection = 0;
}


// ============================================================
// STOP MOTOR
// ============================================================

void stopMotor()
{
    analogWrite(RPWM, 0);
    analogWrite(LPWM, 0);

    digitalWrite(R_DIR, LOW);
    digitalWrite(L_DIR, LOW);

    motorPulseState = false;

    currentDirection = 0;
}


// ============================================================
// START FORWARD PULSE
// ============================================================

void startForwardPulse()
{
    // Direction convention:
    //
    // FORWARD:
    // R = HIGH
    // L = LOW

    digitalWrite(R_DIR, HIGH);
    digitalWrite(L_DIR, LOW);

    analogWrite(RPWM, MOTOR_SPEED);
    analogWrite(LPWM, 0);

    motorPulseState = true;

    lastMotorPulseTime = millis();
}


// ============================================================
// START REVERSE PULSE
// ============================================================

void startReversePulse()
{
    // Direction convention:
    //
    // REVERSE:
    // R = LOW
    // L = HIGH

    digitalWrite(R_DIR, LOW);
    digitalWrite(L_DIR, HIGH);

    analogWrite(RPWM, 0);
    analogWrite(LPWM, MOTOR_SPEED);

    motorPulseState = true;

    lastMotorPulseTime = millis();
}


// ============================================================
// UPDATE DIRECTION
// ============================================================

void updateDirection()
{
    unsigned long now = millis();


    // --------------------------------------------------------
    // Stop requested
    // --------------------------------------------------------

    if (desiredDirection == 0)
    {
        stopMotor();

        reversingSafetyDelay = false;

        return;
    }


    // --------------------------------------------------------
    // Direction change
    // --------------------------------------------------------

    if (currentDirection != 0 &&
        desiredDirection != currentDirection)
    {
        stopMotor();

        reversingSafetyDelay = true;

        reversalStartTime = now;

        return;
    }


    // --------------------------------------------------------
    // Reversal safety delay
    // --------------------------------------------------------

    if (reversingSafetyDelay)
    {
        stopMotor();

        if (now - reversalStartTime >= REVERSAL_STOP_TIME)
        {
            reversingSafetyDelay = false;

            currentDirection = desiredDirection;

            if (currentDirection == 1)
            {
                startForwardPulse();
            }
            else if (currentDirection == -1)
            {
                startReversePulse();
            }
        }

        return;
    }


    // --------------------------------------------------------
    // Start a new direction
    // --------------------------------------------------------

    if (currentDirection == 0)
    {
        currentDirection = desiredDirection;

        if (currentDirection == 1)
        {
            startForwardPulse();
        }
        else if (currentDirection == -1)
        {
            startReversePulse();
        }
    }
}


// ============================================================
// MOTOR BURST / PULSE CONTROL
// ============================================================

void updateMotorPulse()
{
    unsigned long now = millis();


    // Motor is currently ON
    if (motorPulseState)
    {
        if (now - lastMotorPulseTime >= MOTOR_ON_TIME)
        {
            analogWrite(RPWM, 0);
            analogWrite(LPWM, 0);

            motorPulseState = false;

            lastMotorPulseTime = now;
        }

        return;
    }


    // Motor is currently OFF
    if (currentDirection != 0)
    {
        if (now - lastMotorPulseTime >= MOTOR_OFF_TIME)
        {
            if (currentDirection == 1)
            {
                startForwardPulse();
            }
            else if (currentDirection == -1)
            {
                startReversePulse();
            }
        }
    }
}


// ============================================================
// SYSTEM HEARTBEAT
// ============================================================

void systemHeartbeat()
{
    unsigned long now = millis();

    if (now - lastHeartbeat < HEARTBEAT_INTERVAL)
        return;

    lastHeartbeat = now;


    Serial.print("HB ");

    Serial.print(piCommand ? 1 : 0);

    Serial.print(" ");

    Serial.print(
        (now - lastPiCommandTime >= PI_TIMEOUT) ? 1 : 0
    );

    Serial.print(" ");

    Serial.print(currentDirection);

    Serial.print(" ");

    Serial.println(filteredDistance);
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
    // --------------------------------------------------------
    // Serial
    // --------------------------------------------------------

    Serial.begin(115200);


    // --------------------------------------------------------
    // Ultrasonic pins
    // --------------------------------------------------------

    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);

    digitalWrite(TRIG_PIN, LOW);


    // --------------------------------------------------------
    // Motor pins
    // --------------------------------------------------------

    pinMode(RPWM, OUTPUT);
    pinMode(LPWM, OUTPUT);

    pinMode(L_EN, OUTPUT);
    pinMode(R_EN, OUTPUT);

    pinMode(L_DIR, OUTPUT);
    pinMode(R_DIR, OUTPUT);


    // Enable H-bridge
    digitalWrite(L_EN, HIGH);
    digitalWrite(R_EN, HIGH);


    // Start with motor stopped
    stopMotor();


    // --------------------------------------------------------
    // LCD
    // --------------------------------------------------------

    initializeLCD();


    // --------------------------------------------------------
    // HX711
    // --------------------------------------------------------

    scale.begin(HX711_DOUT, HX711_CLK);

    scale.set_scale(calibration_factor);

    scale.tare();


    // --------------------------------------------------------
    // Initialize distance buffer
    // --------------------------------------------------------

    for (int i = 0; i < DISTANCE_BUFFER_SIZE; i++)
    {
        distanceBuffer[i] = 999.0;
    }

    distanceBufferIndex = 0;

    validDistanceSamples = 0;

    filteredDistance = 999.0;

    distance = 999.0;

    distanceValid = false;

    invalidDistanceReadings = 0;


    // --------------------------------------------------------
    // Initialize communication timing
    // --------------------------------------------------------

    lastPiCommandTime = millis();

    lastDistanceTime = millis();

    lastWeightTime = millis();

    lastLCDUpdate = millis();

    lastHeartbeat = millis();

    lastMotorPulseTime = millis();

    lastLCDI2CCheck = millis();

    lastLCDReinit = millis();


    // --------------------------------------------------------
    // Startup LCD
    // --------------------------------------------------------

    lcd.setCursor(0, 0);
    lcd.print("Smart Cart      ");

    lcd.setCursor(0, 1);
    lcd.print("Ready           ");

    delay(1000);

    previousLine1[0] = '\0';
    previousLine2[0] = '\0';
}


// ============================================================
// MAIN LOOP
// ============================================================

void loop()
{
    // --------------------------------------------------------
    // 1. Pi command and watchdog
    // --------------------------------------------------------

    updatePiCommand();


    // --------------------------------------------------------
    // 2. Distance measurement
    // --------------------------------------------------------

    updateDistance();


    // --------------------------------------------------------
    // 3. Load-cell measurement
    // --------------------------------------------------------

    updateWeight();


    // --------------------------------------------------------
    // 4. Determine desired direction
    // --------------------------------------------------------

    determineDirection();


    // --------------------------------------------------------
    // 5. Apply direction
    // --------------------------------------------------------

    updateDirection();


    // --------------------------------------------------------
    // 6. Apply motor pulse control
    // --------------------------------------------------------

    updateMotorPulse();


    // --------------------------------------------------------
    // 7. Update LCD
    // --------------------------------------------------------

    updateLCD();


    // --------------------------------------------------------
    // 8. System heartbeat
    // --------------------------------------------------------

    systemHeartbeat();
}
