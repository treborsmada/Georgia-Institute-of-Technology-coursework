﻿using UnityEngine;
using UnityEngine.SceneManagement;
using System.Collections;

namespace Completed
{
	using System.Collections.Generic;		//Allows us to use Lists.
	using UnityEngine.UI;					//Allows us to use UI.

	public class GameManager : MonoBehaviour
	{

		public static GameManager instance = null;				//Static instance of GameManager which allows it to be accessed by any other script.

		public int level = 1;									//Current level number, expressed in game as "Day 1".
		public int levelsCompleted = 0;
		public int correctAnswers = 0;
		public int incorrectAnswers = 0;
		public int deaths = 0;
		public int score = 0;


		//Awake is always called before any Start functions
		void Awake()
		{
            //Check if instance already exists
            if (instance == null)

                //if not, set instance to this
                instance = this;

            //If instance already exists and it's not this:
            else if (instance != this)

                //Then destroy this. This enforces our singleton pattern, meaning there can only ever be one instance of a GameManager.
                Destroy(gameObject);


			//Sets this to not be destroyed when reloading scene
			DontDestroyOnLoad(gameObject);


		}









	}
}
