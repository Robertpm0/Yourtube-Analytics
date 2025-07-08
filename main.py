import concurrent.futures
import multiprocessing
import streamlit as st
import sqlite3
import requests
import isodate
import concurrent.futures
from typing import List,Tuple
import pandas as pd
import ast
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import streamlit_shadcn_ui as ui
from datetime import datetime
import zipfile
from lxml import etree
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
# set in .sreamlit/secrets.toml
# st.set_page_config(layout="wide")
API_KEY=st.secrets.get("ytv3_key","")  or os.getenv("ytv3_key")
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/videos"

OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "") or os.getenv("OPENROUTER_API_KEY")
TEMP_ZIP_PATH = 'shared_data/uploaded.zip'  # Streamlit will read this


# SQLite Database Configuration
DB_PATH = "ytAnalysis.db"  #used to store duration for a given videokey

# converts seconds integer to days,hours,minutes,seconds and total in hours
def convert_seconds(seconds):
    totHours=seconds //3600
    days = seconds // 86400
    seconds %= 86400
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    
    return days, hours, minutes, seconds,totHours


def get_video_duration_from_db(video_id: str) -> int:
    """
    Checks if a video ID exists in the SQLite database.
    If found, returns its duration; otherwise, returns -1.

    Args:
        video_id (str): The YouTube video ID to check.

    Returns:
        int: The duration in seconds if found, otherwise -1.
    """
    try:
        conn = sqlite3.connect(DB_PATH,timeout=10,check_same_thread=False)
        cursor = conn.cursor()

        # Query the database for the video ID
        cursor.execute("SELECT duration FROM YTvideo WHERE videoID = ?", (video_id,))
        result = cursor.fetchone()

        conn.close()

        return result[0] if result else -1  # Return duration if found, else -1
    except sqlite3.Error as e:
        print(f"SELECT Database error: {e}")
        conn.close()
        return -1  # Return -1 in case of an error

def save_video_duration_to_db(video_id: str, duration: int):
    """
    Saves a video ID and its duration to the SQLite database.

    Args:
        video_id (str): The YouTube video ID.
        duration (int): Duration in seconds.
    """
    try:
        conn = sqlite3.connect(DB_PATH,timeout=10,check_same_thread=False)
        cursor = conn.cursor()

        # Insert the new record
        cursor.execute("INSERT INTO YTvideo (videoID, duration) VALUES (?, ?)", (video_id, duration))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        conn.close()
        print(video_id)
        print(f"INSERT Database error: {e}")

def get_video_durations(video_ids: List[str]) -> Tuple[dict, int]:
    """
    Fetches the duration of YouTube videos (in seconds) for a list of video IDs.
    If a video ID already exists in the database, it uses the stored duration.

    Args:
        video_ids (List[str]): List of YouTube video IDs.

    Returns:
        Tuple[dict, int]: 
            - A dictionary mapping video IDs to their duration in seconds.
            - The number of failed requests.
    """
    durations = {}
    failed_requests = 0  # Track failures
    video_ids_to_fetch = []  # List of IDs that need API calls

    # Step 1: Check database for existing durations
    for video_id in video_ids:
        durations[video_id]=0
        duration = get_video_duration_from_db(video_id)
        if duration != -1:
            # print("Found Existing Duration")
            durations[video_id] = duration  # Use cached value
        else:
            # print(video_id)
            video_ids_to_fetch.append(video_id)  # Add to API fetch list
    # print(len(video_ids_to_fetch))
    # print(len(set(video_ids)))
    # Step 2: Fetch missing durations from YouTube API
    video_id_chunks = [video_ids_to_fetch[i:i + 50] for i in range(0, len(video_ids_to_fetch), 50)]

    def fetch_videos(chunk):
        """Fetch durations for a chunk of video IDs."""
        nonlocal failed_requests
        params = {
            "part": "contentDetails",
            "id": ",".join(chunk),
            "key": API_KEY
        }
        response = requests.get(YOUTUBE_API_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            for item in data.get("items", []):
                video_id = item["id"]
                duration = isodate.parse_duration(item["contentDetails"]["duration"]).total_seconds()
                durations[video_id] = int(duration)

                # Store the new duration in the database
                save_video_duration_to_db(video_id, int(duration))
        else:
            failed_requests += 1  # Increment failure count
            save_video_duration_to_db(video_id,0)
            print(f"Failed request: {response.status_code}, {response.text}")
    # print(d)
    # Step 3: Use ThreadPoolExecutor to parallelize API requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()-1) as executor:
        executor.map(fetch_videos, video_id_chunks)


    # repo = Repo('.')  # if repo is CWD just do '.'

    # repo.index.add(['YTAnalysis.db'])
    # repo.index.commit('System Database Update')
    # origin = repo.remote('origin')
    # origin.push()
    return durations, failed_requests
 # Print first 10 results

# scrapes html input file with watch history, returns collected data in structured fashion
# pass in html file
def parse_json_with_pd(json_content):

    json_content["isPost"]=json_content["titleUrl"].str.contains("post",case=False)
    json_content["hasChannel"]=json_content["subtitles"].str.contains("name",case=False)
    json_content['subtitles'].fillna(' ',inplace=True)

    json_content["details"].fillna(' ',inplace=True)
    ads=[]
    for title in json_content["details"]:
        # print(title[0])
        if title[0]!=' ':
            if 'Ads' in title[0]['name']:
                # titleName=title[0]["name"]
                titleName=True
                ads.append(titleName)


            # print(titleName)
        else:
            titleName=False

            ads.append(titleName)
    json_content["isAd"]=ads




    channels=[]
    for title in json_content["subtitles"]:
        # print(title[0])
        if "name" in title[0]:
            titleName=title[0]["name"]
            # print(titleName)
        else:
            titleName='ad'

        channels.append(titleName)
    

    titles=[]
    for title in json_content["title"]:
        # print(title[0])
        if "Viewed" in title:
            titleName=title.split("Viewed")[1]
        elif "Watched" in title:
            titleName=title.split("Watched")[1]
        # print(titleName)
        titles.append(titleName)
    json_content["Title"]=titles
    # json_content['Title'] = json_content['title'].apply(lambda s: s.split('Viewed')[1] if 'Viewed' in s else s.split('Watched')[1])
    # print(json_content["Title"])
    json_content["Channel"]=channels
    # json_content['Channel'] = json_content['subtitles'].apply(lambda s: s.split("'name':")[0].split(' ')[0] if 'name' in s else 'ad')
    # print(json_content["Channel"])
    keys=[]
    for key in json_content["titleUrl"]:
        try:
            videoKey = key.split("=")[1]

        except:
            try:
                videoKey=key.split("https://www.youtube.com/post/")[0]

            except:
                print("NO KEY FOUND")
                if key=="https://www.youtube.com/watch?v=":
                    videoKey="Deleted"
                # videoKey='NOKEY'
                else:
                    videoKey=""
        keys.append(videoKey)
    json_content["Key"]=keys

    # multiWatch=json_content.groupby(["titleUrl"]).size().
    # watch_counts = df.groupby(['month','isShort']).size().reset_index(name='count')

    return json_content
        # return videoDates,videoDurations,videoKeys,missedVideos,postsLiked,totalVideosWatched,titles,channels,ads,multiWatch,multiChannel,watchHistory

def parse_html_with_lxml(html_content):
        
        parser = etree.HTMLParser()
        tree = etree.fromstring(html_content, parser)

        main_div = tree.xpath('//div[contains(@class, "mdl-grid")]')[0]  # Find main div
        nested_divs = main_div.xpath('./div')  # Get direct child divs
        totalVideosWatched = len(nested_divs)
        
        # st.text(f"Total videos watched: {totalVideosWatched}")
        
        watchHistory = pd.DataFrame()
        adsBool=[]
        postsBool=[]

        vidTitles=[]
        vidKeys=[]
        channelNames=[]
        watchDates=[]

        titles=[]
        channels=[]
        multiChannel={}
        multiWatch={}
        videoDates, videoDurations, videoKeys = [], [], []
        missedVideos = 0
        ads=0
        postsLiked=0
        for div in nested_divs:
            innerContent = div.xpath('.//div[contains(@class, "mdl-grid")]')
            # if there is np inner content skip we cant collect any data, bad entry
            if not innerContent:
                missedVideos += 1
                continue
            
            dataMembers = innerContent[0].xpath('./div')
            # bad entry, skip
            if len(dataMembers) < 2:
                missedVideos += 1
                continue
            
            videoTitleDiv = dataMembers[1]
            isAdDiv=dataMembers[3]
            isAd=False
            # try:
            #     # print()
            #     if "Details" in isAdDiv.xpath('.//b')[1].text.strip():
            #         # isAd=True
            #         ads+=1
            #         # print("AD WATCHES")
            # except:
            #     pass
            # print(videoTitleDiv.get)
            videoUrlElem = videoTitleDiv.xpath('.//a')
            # no video url, invalid video/entry skip
            if not videoUrlElem:
                missedVideos += 1
                continue
            
            videoUrl = videoUrlElem[0].get("href")
            title=videoUrlElem[0].text.strip()
            # get channel for video
            try:
                channel=videoUrlElem[1].text.strip()

            except:
                # if no channel this is an ad
                isAd=True
                ads+=1
                channel='ad'
            # print(title)
            # if this is the video url the video was deleted 
            if title=='https://www.youtube.com/watch?v=':
                title='Deleted Video/Ad'

            dateSibling = videoTitleDiv.xpath('.//br')[-1]
            date = dateSibling.tail.strip()
            isPost=False
            # if there is a video key it will run this block
            # if not it will be a post not a video
            try:
                videoKey = videoUrl.split("=")[1]
            except:
                postsLiked+=1

                videoKey=videoUrl.split("https://www.youtube.com/post/")[0]
                videoKeys.append(videoKey)
                titles.append(title)
                # print(videoUrl)
                isPost=True
                # print(channel)
                videoDates.append(convert_to_date(date))
                channels.append(channel)
            try:
                if videoKey in videoKeys: # when someone views a vid more than once
                    # tracking number of times a channel was watched and specific videos
                    if videoKey not in multiWatch:
                        # print("key",videoKey)
                        multiWatch[videoKey]=1
                        multiChannel[channel]=1
                    else:
                        multiWatch[videoKey]+=1
                        multiChannel[channel]+=1
                else:
                    prettyDate = convert_to_date(date)
                    videoKeys.append(videoKey)
                    videoDates.append(prettyDate)
                    titles.append(title)
                    channels.append(channel)
            except:
                pass
            # store collected data
            adsBool.append(isAd)
            channelNames.append(channel)
            postsBool.append(isPost)
            vidKeys.append(videoKey)
            watchDates.append(convert_to_date(date))
            vidTitles.append(title)

                
        # compressing collected data to a dict
        watchHistory["isAd"]=adsBool
        watchHistory["isPost"]=postsBool
        watchHistory["Title"]=vidTitles
        watchHistory["Date"]=watchDates
        watchHistory["Key"]=vidKeys 
        watchHistory["Channel"]=channelNames
        return videoDates,videoDurations,videoKeys,missedVideos,postsLiked,totalVideosWatched,titles,channels,ads,multiWatch,multiChannel,watchHistory


def convert_to_date(datetime_str):
        """
        Converts a datetime string in the format "Apr 16, 2024, 10:08:37 AM PST" 
        to just the date "YYYY-MM-DD".

        Args:
        - datetime_str (str): The input datetime string.

        Returns:
        - str: The extracted date in "YYYY-MM-DD" format.
        """
        # Define the format (ignoring the timezone)
        date_obj = datetime.strptime(datetime_str[:-4], "%b %d, %Y, %I:%M:%S %p")
        
        # Convert to date string
        return date_obj.strftime("%Y-%m-%d")



# inits the llm used for parsing watch history dataframe
def createDocumentAgent(df):
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-0528-qwen3-8b:free",  # or llama-3, etc.
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )

    agent = create_pandas_dataframe_agent(llm, df,allow_dangerous_code=True,verbose=False)
    return agent





if __name__=="__main__":
    multiprocessing.freeze_support()
    # co
    # initialize globally used datavars
    if 'collected' not in st.session_state:
        st.session_state["collected"]=False
        st.session_state["vidFrame"]=pd.DataFrame()
        st.session_state["vidFrame2"]=pd.DataFrame()
        st.session_state["f"]=""
        st.session_state["comments"]=""
        st.session_state["rep"]=""
        st.session_state["history"]=""
        st.session_state["durs"]=""
        st.session_state["ads"]=""
        st.session_state["subs"]=""
        st.session_state["Playlists"]=""
        st.session_state["list_names"]=""
        st.session_state["all_lists"]=""



    c1,c2,c3=st.columns(3,gap='small')

    # logo image

    st.image("static/yta.svg")
    # link to tutorial
    # ui.link_button(text="How to Use", url="https://docs.google.com/document/d/13R3wwBrTg773rhEE1MFx6w3H1lg4gg-GpiKx72tvrLg/edit?usp=sharing", key="link_btn")


    # st.link_button("How to Use","https://docs.google.com/document/d/13R3wwBrTg773rhEE1MFx6w3H1lg4gg-GpiKx72tvrLg/edit?usp=sharing",icon='❓')
    st.markdown("---")
    st.info("This tool DOES NOT collect your data. Feel free to review our open source codebase to verify our claims.")
    # input data zip file

    # uploaded_file = st.file_uploader("Upload your takeout .zip file", type=["zip"])
    uploaded_file=TEMP_ZIP_PATH = 'shared_data/uploaded.zip'  # Streamlit will read this

    doExperimental=ui.checkbox(label="Get Watchtime (Experimental)")

    # st.session_state["f"]=uploaded_file
    # doExperimental=st.checkbox("Get Watchtime (Experimental)")
    if st.button("visualise",icon='👀'):
        st.session_state["collected"]=False
        st.session_state["vidFrame"]=pd.DataFrame()
        st.session_state["vidFrame2"]=pd.DataFrame()
        st.session_state["f"]=""
        st.session_state["comments"]=""
        st.session_state["rep"]=""
        st.session_state["history"]=""
        st.session_state["durs"]=""
        st.session_state["ads"]=""
        st.session_state["subs"]=""
        st.session_state["Playlists"]=""
        st.session_state["list_names"]=""
        st.session_state["all_lists"]=""
        st.toast("Analyzing Watch History")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Read the uploaded file into a BytesIO buffer
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:

        # with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_ref:
            # List all files in the ZIP
            file_list = zip_ref.namelist()
            #print(file_list)
            # Specify the folder and the HTML file to extract
            target_folder = "Takeout/YouTube and YouTube Music/history/"
            target_html_file = "watch-history.html"  # Change as needed
            target_html_file="watch-history.json"
            target_folder2 = "Takeout/YouTube and YouTube Music/comments/"
            target_folder3 = "Takeout/YouTube and YouTube Music/subscriptions/"


            target_html_file2='comments.csv'
            target_html_file3='subscriptions.csv'

            
            # Construct the full path of the target HTML file
            full_path = f"{target_folder}{target_html_file}"
            full_path2=f"{target_folder2}{target_html_file2}"
            full_path3=f"{target_folder3}{target_html_file3}"
            commentsFiles=[f for f in zip_ref.namelist() if 'comments' in f and '.csv' in f]
            commentsdfs=[]
            if full_path2 in file_list:
                for commentdf in commentsFiles:
                    with zip_ref.open(commentdf) as file:
                        df=pd.read_csv(file)
                        commentsdfs.append(df)
                
                st.session_state["comments"]=pd.concat(commentsdfs,ignore_index=True)
                st.session_state["comments"]["Date"]=pd.to_datetime(st.session_state["comments"]["Comment Create Timestamp"])

                # with zip_ref.open(full_path2,'r') as csv_file:
                #     commentsDf=pd.read_csv(csv_file)
                #     commentsDf["Date"]=pd.to_datetime(commentsDf["Comment Create Timestamp"])
                #     st.session_state["comments"]=commentsDf

            # full_path = f"{target_folder}{target_html_file}"
            # full_path2=f"{target_folder2}{target_html_file2}"
            if full_path3 in file_list:
                with zip_ref.open(full_path3,'r') as csv_file:
                    subsDf=pd.read_csv(csv_file)
                    # subsDf["Date"]=pd.to_datetime(subsDf["Comment Create Timestamp"])
                    st.session_state["subs"]=subsDf


          
            # List of files in the folder, filtering CSVs that aren't the excluded file
            csv_files = [f for f in zip_ref.namelist() if 'playlists' in f and '.csv' in f and 'playlists.csv' not in f]
            # print(csv_files)
            # Read and parse each CSV file
            dataframes = []
            for csv_file in csv_files:
                with zip_ref.open(csv_file) as file:
                    df = pd.read_csv(file)
                    dataframes.append(df)
            st.session_state["Playlists"]=dataframes
            st.session_state["list_names"]=[f.replace(".csv",'') for f in csv_files]
        # Now `dataframes` is a list of DataFrames from all parsed CSVs
        # Optionally combine them into a single DataFrame
            try:
                combined_df = pd.concat(dataframes, ignore_index=True)
                good=True
            except:
                # st.rerun()
                st.error("Error!: Incorrect file uploaded to tool. See bottom of document from the 'How To' Button above for what your file should contain! ")
                # st.rerun()
                good=False
            # if combined_df:
            if good==False:
                st.stop()


            st.session_state['all_lists']=combined_df

        #     # Example: print the combined DataFrame
        #     print(combined_df)
            #print(full_path)
            if full_path in file_list:

                # print("yo")
                # Extract and read the HTML file
                with zip_ref.open(full_path,'r') as html_file:
                    df=pd.read_json(html_file)
                    # print(df.head())
                    # df.to_csv("json22csv1.csv",index=False)
                    # print("dec")
                    # # Normalize the JSON structure
                    # df = pd.json_normalize(html_file,
                    #    meta=['header', 'title', 'titleUrl', 'time', 'products', 'activityControls'],
                    #    errors='ignore')
                    # df.to_csv("josncsv.csv",index=False)

# # Handle entries that do not have "subtitles"
#                     df_no_subtitles = pd.json_normalize(
#     [entry for entry in html_file if 'subtitles' not in entry],
#     sep='_'
# )

# # Combine both into one DataFrame
#                     final_df = pd.concat([df, df_no_subtitles], ignore_index=True, sort=False)
#                     final_df.to_csv("json2csv2.csv",index=False)
# # Display
# print(final_df.head())
                    # html_content = html_file.read().decode("utf-8")

                # df=pd.read_html(html_content)

                # Parse with BeautifulSoup
                # print("got")
                # print(len(df))
                    # df=pd.read_json()
                parsedJson=parse_json_with_pd(df)  
                # videoDates,videoDurations,videoKeys,missedVideos,postsLiked,tot,titles,channels,numAds,repeats,repeatChannel,history=parse_html_with_lxml(html_content)
                # st.session_state["vidFrame"]["WatchDate"]=videoDates
                # st.session_state["vidFrame"]["VideoKey"]=videoKeys
                # st.session_state["vidFrame"]["VideoTitle"]=titles
                # st.session_state.f=repeatChannel
                # st.session_state.reps=repeats
                # st.session_state["vidFrame"]["Channel"]=channels


                # soup = BeautifulSoup(html_content, "html.parser")
                # # print("jj")
                # main_div = soup.find("div", {"class": "mdl-grid"})  # Find div with id="main"
                # # print("prepping")
                # # Find all divs within the main div
                # nested_divs = main_div.find_all("div",recursive=False)
                # totalVideosWatched=len(nested_divs)
                # st.text(fr"toal vids watched: {totalVideosWatched}")
                # watchHistory=pd.DataFrame()
                # # Print all nested divs
                # videoDates=[]
                # videoDurations=[]
                # videoKeys=[]
                # missedVideos=0
                # for div in nested_divs:
                #     innerContent=div.find("div",{"class":"mdl-grid"})
                #     dataMembers=innerContent.find_all("div",recursive=False)
                #     videoTitleDiv=dataMembers[1]
                #     videoUrl=videoTitleDiv.find_all("a",recursive=False)[0].get("href")
                #     videoKey=videoUrl.split("=")[1]
                #     dateSibling=videoTitleDiv.find_all("br",recusive=False)[-1]
                #     date=dateSibling.next_sibling.strip()
                #     prettyDate=convert_to_date(date)
                #     # print(prettyDate)
                #     # print(videoKey)

                #     # print(getYTVideoDuration("VeNfHj6MhgA"))
                #     #videoDuration=getYTVideoDuration(videoKey)
                #     # if videoDuration==0:
                #     #     missedVideos+=1
                #     videoKeys.append(videoKey)
                #     videoDates.append(prettyDate)
                #    # videoDurations.append(videoDuration)
                #     # st.text(rf"{videoUrl}")
                #     # print("\nNested Div:", div)
                #     # print("Div Class:", div.get("class"))
                #

                st.toast("Starting Video Duration Collection") 

                #inc=100/tot
                # import numpy as np
                # inc = np.linspace(0,1,tot)
                #inc =list(range(0,tot))
                #st.text(fr"{len(inc)}")
                # global mBar
                # print(inc)
                history=parsedJson
# Example usage:
                # video_ids = ["VIDEO_ID_1", "VIDEO_ID_2", ..., "VIDEO_ID_30000"]  # Replace with actual video IDs
                if doExperimental==True:

                    videoDurations,mv = get_video_durations(history["Key"])
                else:
                    videoDurations=[0]*len(history)
                # print(len(videoKeys))
                # print(len((videoKeys))))
                # Print a sample result
                # print(list(videoDurations.keys())[:10]) 
                # print(list(videoKeys)[:10]) 
                # print(videoDurations)
                # st.session_state["vidFrame"]["Duration"]=list(videoDurations.values())
                st.session_state["durs"]=videoDurations

                # with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                #     #mBar=st.progress(0.0,"")
                #     # st.session_state["progressBar"]=st.progress(0.0,"Video Duration Collection Process!")
                #     videoDurations,badVids = list(zip(executor.map(getYTVideoDuration, videoKeys,inc,chunksize=1000)))
                # #print()
                # st.session_state["vidFrame2"]=st.session_state["vidFrame"].copy()
                if doExperimental==True:
                    history['Duration'] = history['Key'].map(videoDurations).fillna(0).astype(int)
                else:
                    history["durs"]=[0]*len(history)
                    history['Duration']=history["durs"]
                history['Date']=pd.to_datetime(history["time"],format='ISO8601').dt.date
                # print(history['Date'])
                h1Count=len(history)

                history=history[history["Duration"]<90000]
                h2Count=len(history)
                # history.to_csv("yo.csv",index=False)
                st.session_state.history=history
                st.session_state.collected=True
    if st.session_state.collected==True:
        tabControl=ui.tabs(["Watch History","Comments","Subsctripions & Playlists","AI"])
        view_mode = st.sidebar.radio("Group videos watched by:", ["Month", "Year"])
        st.sidebar.subheader("Adjustments")
        top_n = st.sidebar.slider("Select number of top values to display:", min_value=1, max_value=20, value=10)
        
        min_date=st.session_state.history["Date"].min()
        max_date=st.session_state.history["Date"].max()

        start_date, end_date = st.sidebar.date_input(

            "Select date range:",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        # st.session_state["vidFrame"]["WatchDate"]=pd.to_datetime(st.session_state["vidFrame"]["WatchDate"]).dt.date
        # st.session_state.history["Date"]=pd.to_datetime(history["Date"]).dt.date
        firstRow=st.columns(2)
        if tabControl=="Watch History":
            st. markdown("---")
            st.header("Watch History")

            # with firstRow[0]:
            with st.container(border=True):



        # top videos watched
                df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                df=df[(df['isAd']==False) & (df["isPost"]==False)]
                video_counts = (
            df.groupby(["Key", "Title"])
            .size()
            .reset_index(name="watch_count")
            .sort_values(by="watch_count", ascending=False)
            .head(top_n)
        )
                st.subheader(f"Top {top_n} Videos Watched")
                st.bar_chart(video_counts.set_index("Title")["watch_count"],color="#4AB7FF",y_label="Times Watched",x_label="Video Name")



    # top ads watched 
            # with firstRow[1]:
            with st.container(border=True):
                df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                df=df[df['isAd']==True]
                # print(df.head())
                # df.to_csv("adscsv.csv",index=False)
                video_counts = (
            df.groupby(["Key", "Title"])
            .size()
            .reset_index(name="watch_count")
            .sort_values(by="watch_count", ascending=False)
            .head(top_n)
        )

                st.subheader(f"Top {top_n} Ads Watched")
                st.bar_chart(video_counts.set_index("Title")["watch_count"],color="#E97D62",x_label="Ad Name",y_label="Times Watched")

        # top channels watched

                df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
                df=df[(df['isAd']==False) & (df["isPost"]==False)]
                video_counts = (
            df.groupby(["Channel"])
            .size()
            .reset_index(name="watch_count")
            .sort_values(by="watch_count", ascending=False)
            .head(top_n)
        )
            with st.container(border=True):

                st.subheader(f"Top {top_n} Channels Watched")
                st.bar_chart(video_counts.set_index("Channel")["watch_count"],color="#77B150",y_label="Times Watched",x_label="Channel Name")


            df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]

            df["Date"]=pd.to_datetime(df["Date"])
            print(df["Date"])
            df=df[df["isPost"]==False]
            ui.metric_card("Videos Watched in Selected Time Period" ,f"{len(df)}")
            # --- Group and Aggregate ---
            with st.container(border=True):

                st.subheader(f"Videos Watched per {view_mode} (Videos vs Ads)")
                df["Keys"]=["Is Advertisement" if x==True else "Not Advertisement" for x in df["isAd"].values]
                if view_mode == "Month":
                    df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                    watch_counts = df.groupby(['month','Keys']).size().reset_index(name='count')
                    watch_counts = watch_counts.sort_values('month')
                    print(watch_counts)
                    st.bar_chart(watch_counts.set_index('month'),y='count',color='Keys')  # can also use .bar_chart()
                else:
                    df['year'] = df['Date'].dt.year
                    watch_counts = df.groupby(['year','Keys']).size().reset_index(name='count')
                    st.bar_chart(watch_counts.set_index('year'),y='count',color='Keys')

                # df=df[df["isPost"]==False]
                # ui.metric_card("Videos Watched in Selected Time Period" ,f"{len(df)}")
                # ui.metric_card()




            df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
            # df.to_csv("savedHistyo.csv",index=False)
            with st.container(border=True):

                df["Date"]=pd.to_datetime(df["Date"])
                timeWatched=0
                isShort=[]
                for t in df["Duration"]:
                    if t!=0 and t<55:
                        isShort.append(True)
                    else:
                        isShort.append(False)
                    timeWatched+=t
                df['isShort']=isShort
                df=df[df['isAd']==False]
                df["Keys"]=["Is a Short" if x==True else "Not a Short" for x in df["isShort"].values]

                st.subheader(f"Videos Watched per {view_mode} (Normal Vs Shorts)")
                if view_mode == "Month":
                    df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                    watch_counts = df.groupby(['month','Keys']).size().reset_index(name='count')
                    watch_counts = watch_counts.sort_values('month')
                    print(watch_counts)
                    st.bar_chart(watch_counts.set_index('month'),y='count',color='Keys')  # can also use .bar_chart()
                else:
                    df['year'] = df['Date'].dt.year
                    watch_counts = df.groupby(['year','Keys']).size().reset_index(name='count')
                    st.bar_chart(watch_counts.set_index('year'),y='count',color='Keys')


            df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
            # df.to_csv("savedHistyo.csv",index=False)
            df["Date"]=pd.to_datetime(df["Date"])
            timeWatched=0
            isShort=[]
            for t in df["Duration"]:
                if t!=0 and t<55:
                    isShort.append(True)
                else:
                    isShort.append(False)
                timeWatched+=t
            d,h,m,s,totHours=convert_seconds(timeWatched)
            with st.container(border=True):
                if doExperimental==True:
                    st.subheader(f"Total Duration of Videos Watched from {start_date} to {end_date}")
                    c1,c2,c3,c4=st.columns(4)
                    with c1:
                        # st.metric("Days",d)
                        ui.metric_card("Days",d)
                    with c2:
                        # st.metric("Hours",h)
                        ui.metric_card("Hours",h)

                    with c3:
                        # st.metric("Minutes",m)
                        ui.metric_card("Minutes",m)

                    with c4:
                        # st.metric("Seconds",s)
                        ui.metric_card("Seconds",s)


            
            with st.container(border=True):

                df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]

                df["Date"]=pd.to_datetime(df["Date"])
                print(df["Date"])
                df=df[(df['isAd']==False) & (df["isPost"]==False)]
                if doExperimental==True:
                # --- Group and Aggregate ---
                    st.subheader(f"Total Hours Watched per {view_mode} (Shorts + Regular)")

                    if view_mode == "Month":
                        df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                        # watch_counts = df.groupby(['month','isAd']).size().reset_index(name='count')
                        # watch_counts = watch_counts.sort_values('month')
                        watch_time = (
                df.groupby('month')['Duration']
                .sum()
                .rename("watchTotal")
            )
                        print(watch_time)
                        st.bar_chart((watch_time/3600),color="#C054FA")  # can also use .bar_chart()
                    else:
                        df['year'] = df['Date'].dt.year
                        # df['year'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                        # watch_counts = df.groupby(['month','isAd']).size().reset_index(name='count')
                        # watch_counts = watch_counts.sort_values('month')
                        watch_time = (
                df.groupby('year')['Duration']
                .sum()
                .rename("watchTotal")
            )
                        print(watch_counts)
                        st.bar_chart((watch_time/3600),color="#C054FA")  # can also use .bar_chart()


            df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
            # df.to_csv("savedHistyo.csv",index=False)
            df["Date"]=pd.to_datetime(df["Date"])
            timeWatched=0
            isShort=[]
            for t in df["Duration"]:
                if t!=0 and t<55:
                    isShort.append(True)
                else:
                    isShort.append(False)
                timeWatched+=t
            df['isShort']=isShort

            df=df[(df['isAd']==False) & (df["isPost"]==False)]

            # --- Group and Aggregate ---
            if doExperimental==True:
                with st.container(border=True):

                    st.subheader(f"Hours of Short Videos Watched per {view_mode}")

                    if view_mode == "Month":
                        df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                        # watch_counts = df.groupby(['month','isAd']).size().reset_index(name='count')
                        # watch_counts = watch_counts.sort_values('month')
                        watch_time = (
                df[df['isShort']==True].groupby('month')['Duration']
                .sum()
                .rename("watchTotal")
            )
                        print(watch_time)
                        st.bar_chart((watch_time/3600),color="#54FAE9")  # can also use .bar_chart()
                    else:
                        df['year'] = df['Date'].dt.year
                        # df['year'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                        # watch_counts = df.groupby(['month','isAd']).size().reset_index(name='count')
                        # watch_counts = watch_counts.sort_values('month')
                        watch_time = (
                df[df['isShort']==True].groupby('year')['Duration']
                .sum()
                .rename("watchTotal")
            )
                        print(watch_counts)
                        st.bar_chart((watch_time/3600),color="#54FAE9")  # can also use .bar_chart()

            # df2=df.copy()
        elif tabControl=="Comments":

            st.markdown('---')
            st.header("Comments")
            df2 = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
            # df.to_csv("savedHistyo.csv",index=False)
            df2["Date"]=pd.to_datetime(df2["Date"])
            timeWatched=0
            isShort=[]
            for t in df2["Duration"]:
                if t!=0 and t<55:
                    isShort.append(True)
                else:
                    isShort.append(False)
                timeWatched+=t
            df2['isShort']=isShort

            df2=df2[(df2['isAd']==False) & (df2["isPost"]==False)]
            df = st.session_state.comments[(st.session_state.comments["Date"].dt.date >= pd.to_datetime(start_date).date()) & (st.session_state.comments["Date"].dt.date <= pd.to_datetime(end_date).date())]
            compCols=st.columns(2)
            rate=(len(df)/len(df2))*100
            per=100
            if rate <1:
                rate = (len(df)/len(df2))*1000
                per=1000
            if rate <1:
                rate = (len(df)/len(df2))*10000
                per=10000
            with st.container(border=True):

                with compCols[0]:
                    ui.metric_card("Total Comments Left",f"{len(df)}")
                with compCols[1]:
                    ui.metric_card("Comment Frequency",f"{int(rate)} comments per {per} videos")
            with st.container(border=True):

                st.subheader(f"Comments left per {view_mode}")
                # df = st.session_state.comments[(st.session_state.comments["Date"].dt.date >= pd.to_datetime(start_date).date()) & (st.session_state.comments["Date"].dt.date <= pd.to_datetime(end_date).date())]
                df["Date"]=pd.to_datetime(df["Date"])
                if view_mode == "Month":
                    df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                    watch_counts = df.groupby(['month']).size().reset_index(name='count')
                    watch_counts = watch_counts.sort_values('month')
                    print(watch_counts)
                    st.bar_chart(watch_counts.set_index('month'),color="#61256F")  # can also use .bar_chart() 
                else:
                    df['year'] = df['Date'].dt.year
                    watch_counts = df.groupby(['year']).size().reset_index(name='count')
                    st.bar_chart(watch_counts.set_index('year'),color="#61256F")

            def extract_text(entry):
                try:
                    data = ast.literal_eval(entry)  # Safely convert string to dict
                    return data.get("text", "")  # Remove asterisks
                except:
                    return ""

            df['clean_text'] = df['Comment Text'].apply(extract_text)
            with st.container(border=True):

                st.subheader('Comments Word Cloud')
                # Step 2: Create a word cloud
                all_text = ' '.join(df['clean_text'].dropna())
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title("Word Cloud of Comments")
                st.pyplot(plt.gcf())
                # plt.show()

            # Step 3: Sentiment analysis using TextBlob
            def get_sentiment(text):
                return TextBlob(text).sentiment.polarity  # Range: -1 (negative) to 1 (positive)
            with st.container(border=True):

                st.subheader(f'Average Sentiment of Comments per {view_mode}')
                st.caption("Negative Sentiment indicate meaner comments, positive indicates not as mean comments")
                df['sentiment'] = df['clean_text'].apply(get_sentiment)
                if view_mode == "Month":
                    df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                    watch_counts = df.groupby(['month'])['sentiment'].mean().reset_index(name='count')
                    watch_counts = watch_counts.sort_values('month')
                    # print(watch_counts)
                    st.bar_chart(watch_counts.set_index('month'),color="#C4ED3B")  # can also use .bar_chart() 
                else:
                    df['year'] = df['Date'].dt.year
                    watch_counts = df.groupby(['year'])['sentiment'].mean().reset_index(name='count')
                    st.bar_chart(watch_counts.set_index('year'),color="#C4ED3B")
        elif tabControl=="Subsctripions & Playlists":
            st.markdown('---')

                # st.subheader(f"Total Number of Subscriptions: {len(st.session_state['subs'])}")
            st.header("Playlists & Subscriptions")
            with st.container(border=True):

                stats=st.columns(3)
                with stats[0]:
                    ui.metric_card("Total Number of Subscriptions",f"{len(st.session_state['subs'])}")
                with stats[1]:
                    ui.metric_card(f"Total Number of Playlists", f"{(len(st.session_state['Playlists']))}")
                # st.subheader(f"Total Number of Playlists {(len(st.session_state['Playlists']))}")
                with stats[2]:
                    ui.metric_card(f"Total Videos in All Playlists",f"{len(st.session_state['all_lists'])}")
                # st.subheader(f'Total Videos in All Playlists {len(st.session_state["all_lists"])}')
            lengths=[]
            i=0

            for x in st.session_state['list_names']:
                # if x ==st.session_state['list_names'][0]:
                #     i+=1
                #     continue
                # else:
                lengths.append(len(st.session_state['Playlists'][i]))
                i+=1
            with st.container(border=True):

                st.subheader("Videos per Playlist")
                st.session_state['list_names']=[g.replace("Takeout/YouTube and YouTube Music/playlists/","") for g in st.session_state['list_names']]
                st.bar_chart(pd.DataFrame({"Playlist":st.session_state['list_names'],'Number of Videos':lengths}),x='Playlist',y='Number of Videos',color="#4DD799")
            
            df=st.session_state['all_lists']
            # df = st.session_state.history[((pd.to_datetime(stjj.session_state['all_lists']["Playlist Video Creation Timestamp"])).dt.to_period('M').dt.to_timestamp() >= pd.to_datetime(start_date).date()) & (pd.to_datetime(st.session_state['all_lists']["Playlist Video Creation Timestamp"]) <= pd.to_datetime(end_date).date())]

            df["Date"]=pd.to_datetime(df['Playlist Video Creation Timestamp'])
            df["Date2"]=df["Date"].dt.date
            df=df[(df["Date2"]>=pd.to_datetime(start_date).date()) & (df["Date2"]<=pd.to_datetime(end_date).date())]
            # print(df["month"])
            with st.container(border=True):

                st.subheader("Videos Added to Playlist per Month")
                if view_mode == "Month":
                    df['month'] = df['Date'].dt.to_period('M').dt.to_timestamp()  # first day of each month
                    watch_counts = df.groupby(['month']).size().reset_index(name='count')
                    watch_counts = watch_counts.sort_values('month')
                    # print(watch_counts)
                    st.bar_chart(watch_counts.set_index('month'),color="#FF1988")  # can also use .bar_chart() 
                else:
                    df['year'] = df['Date'].dt.year
                    watch_counts = df.groupby(['year']).size().reset_index(name='count')
                    st.bar_chart(watch_counts.set_index('year'),color="#FF1988")
      
                # btn(username="robertmundo", floating=False, width=100)\
        elif tabControl=="AI":
            df = st.session_state.history[(st.session_state.history["Date"] >= pd.to_datetime(start_date).date()) & (st.session_state.history["Date"] <= pd.to_datetime(end_date).date())]
            st.markdown('---')

            with st.container(border=True):

                st.subheader("Chat with your Data!")
                st.badge("AI",icon='🤖',color="violet",width='stretch')
                
                agent=createDocumentAgent(df)
                # agent = create_pandas_dataframe_agent(llm, df, allow_dangerous_code=True,verbose=False)

                user_input = st.text_input("Ask About Your Watch History",placeholder="Enter question")

                if user_input:
                    with st.spinner("Thinking..."):
                        try:
                            response = agent.run(user_input)
                            st.session_state.chat_history.append(("You", user_input))
                            st.session_state.chat_history.append(("WatchBot", response))
                        except Exception as e:
                            st.error(f"Error: {e}")

                # Display chat history
                for sender, message in st.session_state.chat_history:
                    st.markdown(f"**{sender}:** {message}")

    footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {

left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with 💖 by <a style='display: block; text-align: center;' href="https://www.robertmundo.netlify.app/" target="_blank">RPM III</a></p>
<style>.bmc-button img{width: 27px !important;margin-bottom: 1px !important;box-shadow: none !important;border: none !important;vertical-align: middle !important;}.bmc-button{line-height: 36px !important;height:37px !important;text-decoration: none !important;display:inline-flex !important;color:#ffffff !important;background-color:#FF813F !important;border-radius: 3px !important;border: 1px solid transparent !important;padding: 1px 9px !important;font-size: 23px !important;letter-spacing: 0.6px !important;box-shadow: 0px 1px 2px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;margin: 0 auto !important;font-family:'Cookie', cursive !important;-webkit-box-sizing: border-box !important;box-sizing: border-box !important;-o-transition: 0.3s all linear !important;-webkit-transition: 0.3s all linear !important;-moz-transition: 0.3s all linear !important;-ms-transition: 0.3s all linear !important;transition: 0.3s all linear !important;}.bmc-button:hover, .bmc-button:active, .bmc-button:focus {-webkit-box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;text-decoration: none !important;box-shadow: 0px 1px 2px 2px rgba(190, 190, 190, 0.5) !important;opacity: 0.85 !important;color:#ffffff !important;}</style><link href="https://fonts.googleapis.com/css?family=Cookie" rel="stylesheet"><a class="bmc-button" target="_blank" href="https://www.buymeacoffee.com/robertmundo"><img src="https://www.buymeacoffee.com/assets/img/BMC-btn-logo.svg" alt="Buy me a coffee"><span style="margin-left:5px">Buy me a coffee</span></a>
</div>
"""
    st.markdown(footer,unsafe_allow_html=True)

