# Speech-Emotion-Recognition

a program that takes a sound file contains speech and analyzie the emotions of the speeker

## Group Members:

1. Tahany Ali
2. Tasneem Al-Absi
3. Abdullah Nazzal
4. Anas Abusaif


## Trello board

[Trello-board](https://trello.com/b/gnFGr9Nu/speech-emotion-recognition)

## User Stories:

1- As a user I would want the program to be able to extract sound features that I can enter any files I want.

Feature Tasks:

- User can choose sound files to be extract.

Acceptance Tests:

- given : user not enter the sound file yet.
- when : user will choose sound from its path correctly.
- then : the file is read correctly without errors.

Acceptance Tests:

- given : user not enter the sound file yet  .
- when : user choose the wrong sound file path.
- then : an error message shown to them.

Estimates Time:

- 4 Hours

***

2- As a user I want to see The emotion of a certain speaker is extracted from a sound file and shown so that I can know how a certain person feels.

Feature Tasks:

- User can see the emotion resulted from the input of speech.

Acceptance Tests:

- given: user entered the sound file and waiting for the results .
- when : they enter the path of sound file they want to analyze .
- then :  the emotion related to speech is recognized correctly.

Acceptance Tests:

- given: user entered the sound file and waiting for the results .
- when : they enter the path of sound file they want to analyze .
- then :  the emotion related to speech is recognized not correctly.

Estimates Time:

- 4 Hours

***

3- As a user I want to to see the accuracy of the resulted emotion so that I can know if the application works correctly.

Feature Tasks:

- User can see the accuracy of the model.

Acceptance Tests:

- given: user has seen the result.
- when : they order to see how correct the result is.
- then :  the accuracy is shown to the user.

Acceptance Tests:

- given: user has seen the result.
- when : they order to see how correct the result is.
- then : show message to try with another clear sound file if the accuracy value is low.

Estimates Time:

- 4 Hours

***

4- As a user I want to see a visual representation of the sound waves of the sound file so that I can see how the sound transmits through the air.

Feature Tasks:

- User can see the visual representation of sound file .

Acceptance Tests:

- given: user has enter of sound file correctly.
- when : they order from command to print the plot of the sound waves from a specific sound file.
- then: the visual representation will be shown correctly as expected.

Estimates Time:

- 4 Hours

***
5- As a user I want to analyze the change of the mood for a person during the day so that I can suggest a book for them.

Feature Tasks:

- Ability to extract emotions from a number of files and analyze the set of emotions.

Acceptance Tests:

- given: user has seen the accuracy and app works fine. 
- when : they insert a number of sound files for the same person.
- then: all emotions that related to these files will be extracted and shown the result.

Acceptance Tests:

- given: user has seen the accuracy and app works fine. 
- when : they insert a number of sound files for the same person.
- then: all emotions that related to these files will be extracted and shown the result.

Estimates Time:

- 5 Hours

***


## Domain Modeling
[domain](https://miro.com/app/board/o9J_llx5lAA=/?invite_link_id=407537396113)

## Data Set:
[RAVDESS](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)
