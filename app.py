import json
import time, sys,  os ,random
from pyspark import SparkContext, SparkConf

from recommend import RecommendationEngine
  
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
'''@main.route("/<int:user_id>/ratings", methods = ["POST"])'''
'''history_str str,str'''
def add_ratings(user_id,history_str):
    # get the ratings from the Flask POST request object
    ratings_list = map(lambda x: x.split(","), history_str)
    # create a list with the format required by the negine (user_id, gj_resume_id, rating)
    ratings = map(lambda x: (user_id, int(x[0]), float(x[1])), ratings_list)
    # add them to the model using then engine API
    recommendation_engine.add_ratings(ratings)
 
    return json.dumps(ratings)
 
 
def create_app(spark_context, dataset_path):
    global recommendation_engine 
 
    recommendation_engine = RecommendationEngine(spark_context, dataset_path)    
    


def init_spark_context():
    ## load spark context
    conf = SparkConf().setAppName("gj_resume_recommendation-server")
    ## IMPORTANT: pass aditional Python modules to each worker
    sc = SparkContext(conf=conf, pyFiles=['recommend.py'])
 
    return sc



if __name__ == "__main__":
	# Init spark context and load libraries
	sc = init_spark_context()
	dataset_path = os.path.join('datasets', 'ml-latest')
	app = create_app(sc, dataset_path)
 

	for i in range(5):

		user_id=random.randrange(1, 466, 3);
		logger.info("==============>")
		logger.info(user_id)

		logger.info("start top N... N=5")
		### top N
		count=2
		top_ratings = recommendation_engine.get_top_ratings(user_id,count)
		logger.info(top_ratings)


		#### user +gj_resume- --->score
		#user <1,466>
		logger.info("start find user xxx in moive rating ")

		gj_resume_ids=[155820,25995,8965]


		ratings = recommendation_engine.get_ratings_for_gj_resume_ids(user_id, gj_resume_ids)
		str=  json.dumps(ratings)
		logger.info(str)


