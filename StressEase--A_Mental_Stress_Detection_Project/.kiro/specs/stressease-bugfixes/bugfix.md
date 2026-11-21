# Bugfix Requirements Document

## Introduction

This document outlines the bugfix requirements for two critical issues in the StressEase mental health monitoring application:

1. **Email Validation Bug**: The signup and login system accepts invalid email formats, allowing users to register and authenticate with improperly formatted email addresses.
2. **Face Detection Feature Not Working**: The facial expression detection feature is non-functional due to missing OpenCV dependency, preventing users from accessing real-time stress monitoring via webcam.

These bugs impact core functionality - authentication security and a primary feature of the stress monitoring system.

## Bug Analysis

### Bug 1: Email Validation

#### Current Behavior (Defect)

1.1 WHEN a user enters a string without "@" symbol (e.g., "abc", "test", "user") in the email field during signup THEN the system accepts it as a valid email and creates an account

1.2 WHEN a user enters a string with "@" but no domain (e.g., "user@", "@domain") in the email field during signup THEN the system accepts it as a valid email and creates an account

1.3 WHEN a user enters a string with invalid domain structure (e.g., "user@domain", "test@d") in the email field during signup THEN the system accepts it as a valid email and creates an account

1.4 WHEN a user enters an invalid email format during login THEN the system attempts to authenticate against the invalid email stored in the database

#### Expected Behavior (Correct)

2.1 WHEN a user enters a string without "@" symbol in the email field during signup THEN the system SHALL reject the input and display an error message "Invalid email format. Please enter a valid email address."

2.2 WHEN a user enters a string with "@" but no domain or invalid domain structure in the email field during signup THEN the system SHALL reject the input and display an error message "Invalid email format. Please enter a valid email address."

2.3 WHEN a user enters a valid email format (contains @ and valid domain like user@example.com) during signup THEN the system SHALL accept the email and proceed with account creation

2.4 WHEN a user enters an invalid email format during login THEN the system SHALL reject the input before attempting database authentication and display an error message "Invalid email format."

#### Unchanged Behavior (Regression Prevention)

3.1 WHEN a user enters a valid email address that already exists in the database during signup THEN the system SHALL CONTINUE TO display "Account already exists. Please login."

3.2 WHEN a user enters valid credentials (valid email format + correct password) during login THEN the system SHALL CONTINUE TO authenticate successfully and redirect to home page

3.3 WHEN a user enters valid email format but incorrect password during login THEN the system SHALL CONTINUE TO display "Invalid email or password."

3.4 WHEN the email is stored in the database after successful signup THEN the system SHALL CONTINUE TO store it in lowercase format

### Bug 2: Face Detection Not Working

#### Current Behavior (Defect)

1.1 WHEN a user navigates to the facial expression detection page (/face-detection route) THEN the page loads but the face analysis feature is non-functional

1.2 WHEN the application attempts to analyze a webcam frame via the /api/analyze-face endpoint THEN the system returns error response {"error": "OpenCV is not installed in this environment."} with status code 503

1.3 WHEN the application code executes the face detection module on startup THEN cv2 is set to None and a warning "OpenCV unavailable. Facial expression detection disabled." is logged

1.4 WHEN a user attempts to use the real-time stress monitoring feature that depends on facial expression analysis THEN the feature fails silently or returns errors to the frontend

#### Expected Behavior (Correct)

2.1 WHEN a user navigates to the facial expression detection page (/face-detection route) THEN the page SHALL load and the face analysis feature SHALL be fully functional

2.2 WHEN the application receives a valid webcam frame via the /api/analyze-face endpoint THEN the system SHALL analyze the frame using OpenCV and return emotion detection results (Happy, Tense, Neutral) with stress signal levels (Low, Moderate, Mild) and confidence scores

2.3 WHEN the application code executes the face detection module on startup THEN cv2 SHALL be successfully imported and Haar cascade classifiers SHALL be loaded without errors

2.4 WHEN OpenCV is not installed in the environment THEN the system SHALL log a clear error message instructing the user to run "pip install opencv-python" to enable the face detection feature

2.5 WHEN a face is not detected in the webcam frame THEN the system SHALL return a user-friendly response with "face_detected": false and recommendation "Keep your face centered in good lighting and try again."

#### Unchanged Behavior (Regression Prevention)

3.1 WHEN the application uses the existing face detection logic with Haar cascade classifiers (for face and smile detection) THEN the system SHALL CONTINUE TO use the same detection algorithms and thresholds

3.2 WHEN OpenCV CascadeClassifier is not available in a particular OpenCV build THEN the system SHALL CONTINUE TO fall back to skin-tone masking and central face-region estimation

3.3 WHEN the /api/fer-model-status endpoint is called THEN the system SHALL CONTINUE TO return the status of the FER2013 CNN model and fallback mechanism

3.4 WHEN face detection results are saved THEN the system SHALL CONTINUE TO store them in both MongoDB (via safe_insert) and SQLite (via save_sqlite_record) with record_type "face_expression"

3.5 WHEN MongoDB is unavailable THEN the system SHALL CONTINUE TO function properly by logging warnings and using SQLite as the fallback database
