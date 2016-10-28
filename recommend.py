#! /bin/env python 
import os
from pyspark.mllib.recommendation import ALS
 
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_counts_and_averages(ID_and_ratings_tuple):
    '''Given a tuple (gj_resumeID, ratings_iterable) 
    returns (gj_resumeID, (ratings_count, ratings_avg))
    '''
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1])) / nratings)



class RecommendationEngine:
	def __count_and_average_ratings(self):
		# Updates the gj_resumes ratings counts from 
		# the current data self.ratings_RDD
		logger.info("Counting gj_resume ratings...")
		gj_resume_ID_with_ratings_RDD = self.ratings_RDD.map(lambda x: (x[1], x[2])).groupByKey()
		gj_resume_ID_with_avg_ratings_RDD = gj_resume_ID_with_ratings_RDD.map(get_counts_and_averages)
		self.gj_resumes_rating_counts_RDD = gj_resume_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))


	def __train_model(self):
		#Train the ALS model with the current dataset
		logger.info("Training the ALS model...")
		self.model = ALS.train(self.ratings_RDD, self.rank, seed=self.seed,
				       iterations=self.iterations, lambda_=self.regularization_parameter)
		logger.info("ALS model built!")


	def __init__(self, sc, dataset_path):
		#Init the recommendation engine given a Spark context and a dataset path

		logger.info("Starting up the Recommendation Engine: ")

		self.sc = sc

		# Load ratings data for later use
		logger.info("Loading Ratings data...")
		ratings_file_path = os.path.join(dataset_path, 'ratings.csv')
		ratings_raw_RDD = self.sc.textFile(ratings_file_path)
		ratings_raw_data_header = ratings_raw_RDD.take(1)[0]
		self.ratings_RDD = ratings_raw_RDD.filter(lambda line: line != ratings_raw_data_header)\
		    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))).cache()
		# Load gj_resumes data for later use
		logger.info("Loading Movies data...")
		gj_resumes_file_path = os.path.join(dataset_path, 'gj_resumes.csv')
		gj_resumes_raw_RDD = self.sc.textFile(gj_resumes_file_path)
		gj_resumes_raw_data_header = gj_resumes_raw_RDD.take(1)[0]
		self.gj_resumes_RDD = gj_resumes_raw_RDD.filter(lambda line: line != gj_resumes_raw_data_header)\
		    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]), tokens[1], tokens[2])).cache()
		self.gj_resumes_titles_RDD = self.gj_resumes_RDD.map(lambda x: (int(x[0]), x[1])).cache()
		# Pre-calculate gj_resumes ratings counts
		self.__count_and_average_ratings()

		# Train the model
		self.rank = 8
		self.seed = 5L
		self.iterations = 10
		self.regularization_parameter = 0.1
		self.__train_model() 





	def add_ratings(self, ratings):
		#Add additional gj_resume ratings in the format (user_id, gj_resume_id, rating)
		# Convert ratings to an RDD
		new_ratings_RDD = self.sc.parallelize(ratings)
		# Add new ratings to the existing ones
		self.ratings_RDD = self.ratings_RDD.union(new_ratings_RDD)
		# Re-compute gj_resume ratings count
		self.__count_and_average_ratings()
		# Re-train the ALS model with the new ratings
		self.__train_model()

		return ratings


	def __predict_ratings(self, user_and_gj_resume_RDD):
		#Gets predictions for a given (userID, gj_resumeID) formatted RDD
		#Returns: an RDD with format (gj_resumeTitle, gj_resumeRating, numRatings)
		predicted_RDD = self.model.predictAll(user_and_gj_resume_RDD)
		predicted_rating_RDD = predicted_RDD.map(lambda x: (x.product, x.rating))
		predicted_rating_title_and_count_RDD = \
		predicted_rating_RDD.join(self.gj_resumes_titles_RDD).join(self.gj_resumes_rating_counts_RDD)
		predicted_rating_title_and_count_RDD = \
		predicted_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

		return predicted_rating_title_and_count_RDD

	def get_top_ratings(self, user_id, gj_resumes_count):
		#Recommends up to gj_resumes_count top unrated gj_resumes to user_id'''

		# Get pairs of (userID, gj_resumeID) for user_id unrated gj_resumes
		user_unrated_gj_resumes_RDD = self.ratings_RDD.filter(lambda rating: not rating[1] == user_id).map(lambda x: (user_id, x[1]))
		# Get predicted ratings
		ratings = self.__predict_ratings(user_unrated_gj_resumes_RDD).filter(lambda r: r[2] >= 25).takeOrdered(gj_resumes_count, key=lambda x:-x[1])

		return ratings



	def get_ratings_for_gj_resume_ids(self, user_id, gj_resume_ids):
		#Given a user_id and a list of gj_resume_ids, predict ratings for them 
		
		requested_gj_resumes_RDD = self.sc.parallelize(gj_resume_ids).map(lambda x: (user_id, x))
		# Get predicted ratings
		ratings = self.__predict_ratings(requested_gj_resumes_RDD).collect()

		return ratings





