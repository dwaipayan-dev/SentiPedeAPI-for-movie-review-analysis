# SentiPedeAPI-for-movie-review-analysis
This is a flask API I made where you can get a score on a given movie review on a scale of 0(Extremely bad) to 10(Extremely good). This API is powered by an NLP model based on LSTM word-based model architecture and supports a word limit of 150 words.  The val AUC of the model is 0.85 and you can find it in my Keras profile. You need only worry about the word limit as the api transforms the input to model appropriate representation

# Routes 
The API has two routes.
1. '/': This route opens up a textarea where you can input a review and on clicking the submit button the application redirects to a page outputting the score.

2. '/predict': Intermediate route used to return the result's view.

3. '/predict-api': This is a route that can be used by developers to send a review via post request. The function mapped to this route returns the score of the review along with the status of the request. Make sure post request body is of type json. Ajax by default sends form type data. So specify content-type as json OR in the function change request.get_json() to request.form.
