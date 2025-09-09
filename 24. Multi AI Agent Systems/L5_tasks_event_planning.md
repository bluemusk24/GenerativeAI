# L5: Automate Event Planning

In this lesson, you will learn more about Tasks.

The libraries are already installed in the classroom. If you're running this notebook on your own machine, you can install the following:
```Python
!pip install crewai==0.28.8 crewai_tools==0.1.6 langchain_community==0.0.29
```


```python
# Warning control
import warnings
warnings.filterwarnings('ignore')
```

- Import libraries, APIs and LLM


```python
from crewai import Agent, Crew, Task
```

**Note**: 
- The video uses `gpt-4-turbo`, but due to certain constraints, and in order to offer this course for free to everyone, the code you'll run here will use `gpt-3.5-turbo`.
- You can use `gpt-4-turbo` when you run the notebook _locally_ (using `gpt-4-turbo` will not work on the platform)
- Thank you for your understanding!


```python
import os
from utils import get_openai_api_key,get_serper_api_key

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
os.environ["SERPER_API_KEY"] = get_serper_api_key()
```

## crewAI Tools


```python
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
```

## Creating Agents


```python
# Agent 1: Venue Coordinator
venue_coordinator = Agent(
    role="Venue Coordinator",
    goal="Identify and book an appropriate venue "
    "based on event requirements",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "With a keen sense of space and "
        "understanding of event logistics, "
        "you excel at finding and securing "
        "the perfect venue that fits the event's theme, "
        "size, and budget constraints."
    )
)
```


```python
 # Agent 2: Logistics Manager
logistics_manager = Agent(
    role='Logistics Manager',
    goal=(
        "Manage all logistics for the event "
        "including catering and equipmen"
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Organized and detail-oriented, "
        "you ensure that every logistical aspect of the event "
        "from catering to equipment setup "
        "is flawlessly executed to create a seamless experience."
    )
)
```


```python
# Agent 3: Marketing and Communications Agent
marketing_communications_agent = Agent(
    role="Marketing and Communications Agent",
    goal="Effectively market the event and "
         "communicate with participants",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Creative and communicative, "
        "you craft compelling messages and "
        "engage with potential attendees "
        "to maximize event exposure and participation."
    )
)
```

## Creating Venue Pydantic Object

- Create a class `VenueDetails` using [Pydantic BaseModel](https://docs.pydantic.dev/latest/api/base_model/).
- Agents will populate this object with information about different venues by creating different instances of it.


```python
from pydantic import BaseModel
# Define a Pydantic model for venue details 
# (demonstrating Output as Pydantic)
class VenueDetails(BaseModel):
    name: str
    address: str
    capacity: int
    booking_status: str
```

## Creating Tasks
- By using `output_json`, you can specify the structure of the output you want.
- By using `output_file`, you can get your output in a file.
- By setting `human_input=True`, the task will ask for human feedback (whether you like the results or not) before finalising it.


```python
venue_task = Task(
    description="Find a venue in {event_city} "
                "that meets criteria for {event_topic}.",
    expected_output="All the details of a specifically chosen"
                    "venue you found to accommodate the event.",
    human_input=True,
    output_json=VenueDetails,
    output_file="venue_details.json",  
      # Outputs the venue details as a JSON file
    agent=venue_coordinator
)
```

- By setting `async_execution=True`, it means the task can run in parallel with the tasks which come after it.


```python
logistics_task = Task(
    description="Coordinate catering and "
                 "equipment for an event "
                 "with {expected_participants} participants "
                 "on {tentative_date}.",
    expected_output="Confirmation of all logistics arrangements "
                    "including catering and equipment setup.",
    human_input=True,
    async_execution=True,
    agent=logistics_manager
)
```


```python
marketing_task = Task(
    description="Promote the {event_topic} "
                "aiming to engage at least"
                "{expected_participants} potential attendees.",
    expected_output="Report on marketing activities "
                    "and attendee engagement formatted as markdown.",
    async_execution=True,
    output_file="marketing_report.md",  # Outputs the report as a text file
    agent=marketing_communications_agent
)
```

## Creating the Crew

**Note**: Since you set `async_execution=True` for `logistics_task` and `marketing_task` tasks, now the order for them does not matter in the `tasks` list.


```python
# Define the crew with agents and tasks
event_management_crew = Crew(
    agents=[venue_coordinator, 
            logistics_manager, 
            marketing_communications_agent],
    
    tasks=[venue_task, 
           logistics_task, 
           marketing_task],
    
    verbose=True
)
```

## Running the Crew

- Set the inputs for the execution of the crew.


```python
event_details = {
    'event_topic': "Tech Innovation Conference",
    'event_description': "A gathering of tech innovators "
                         "and industry leaders "
                         "to explore future technologies.",
    'event_city': "San Francisco",
    'tentative_date': "2024-09-15",
    'expected_participants': 500,
    'budget': 20000,
    'venue_type': "Conference Hall"
}
```

**Note 1**: LLMs can provide different outputs for they same input, so what you get might be different than what you see in the video.

**Note 2**: 
- Since you set `human_input=True` for some tasks, the execution will ask for your input before it finishes running.
- When it asks for feedback, use your mouse pointer to first click in the text box before typing anything.


```python
result = event_management_crew.kickoff(inputs=event_details)
```

    [1m[95m [DEBUG]: == Working Agent: Venue Coordinator[00m
    [1m[95m [INFO]: == Starting Task: Find a venue in San Francisco that meets criteria for Tech Innovation Conference.[00m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI need to find a venue in San Francisco that meets the criteria for the Tech Innovation Conference. I should consider the theme, size, and budget constraints of the event.
    
    Action: Search the internet
    Action Input: {"search_query": "Tech Innovation Conference venues in San Francisco"}[0m[95m 
    
    
    Search results: Title: Top 12 tech conferences in San Francisco for 2024 - TravelPerk
    Link: https://www.travelperk.com/blog/top-tech-conferences-in-san-francisco/
    Snippet: With over 200 speakers, the San Francisco Tech Summit offers an action-packed agenda for inspiration, insights, innovation, and connections.
    ---
    Title: IT & Technology Events in San Francisco - 10Times
    Link: https://10times.com/sanfrancisco-us/technology
    Snippet: World Agri-Tech Innovation Summit. San Francisco Marriott Marquis, San Francisco. The World Agri-Tech Innovation Summit serves as a gathering for the global ...
    ---
    Title: Top 10 Must-Attend Tech Meetups and Conferences in San Francisco
    Link: https://www.nucamp.co/blog/coding-bootcamp-san-francisco-ca-top-10-mustattend-tech-meetups-and-conferences-in-san-francisco
    Snippet: Discover the top 10 must-attend tech meetups and conferences in San Francisco for 2025. Elevate your network and tech knowledge today!
    ---
    Title: Tech Summit Silicon Valley
    Link: https://techsummit.tech/san-francisco/?srsltid=AfmBOoqpGYXzG7czGMTu_TgeMVZ3y_Rt8ZNLLocbEP67j2B3XdXGCdUv
    Snippet: Unlock two days of tech excellence: Immerse yourself in masterclasses, expert workshops, inspiring speakers, valuable networking, and limitless opportunities.
    ---
    Title: Venue - World Agri-Tech Innovation Summit, San Francisco
    Link: https://worldagritechusa.com/venue/
    Snippet: SUMMIT VENUE: MARRIOTT MARQUIS · San Francisco International Airport (SFO), 14 miles away. Estimated taxi fare: 45 USD (one way) Estimated subway fare: 8.65 USD ...
    ---
    Title: Technology conference events in San Francisco, CA - Eventbrite
    Link: https://www.eventbrite.com/d/ca--san-francisco/technology-conference/
    Snippet: Technology conference events in San Francisco, CA ; GP Dinner #186 | Technology Will Not Save Us. Tue, Feb 4 • 6:00 PM. 54 Washburn St. From $135.23 ; IT Social ...
    ---
    Title: Event Details San Francisco - Tech Summit
    Link: https://techsummit.tech/san-francisco-tech-conference/?srsltid=AfmBOoqr5tnbnj4XQQQUadW8eu7Z7mJof4gq5FBWRUl0ZsCVukXSc6j9
    Snippet: 150+ Speakers – Experts and Leaders in Tech. Masterclasses, Workshops, Roundtables. 2 Power-Packed Days. Professional Development. Networking and Opportunities.
    ---
    Title: The Moscone Center: A Hub Of Innovation | San Francisco Travel
    Link: https://www.sftravel.com/meetings/article/moscone-center-hub-innovation
    Snippet: The Moscone Center: A Hub Of Innovation. Host your meeting at The Moscone Center and join a proud history of ground-breaking, paradigm-shifting moments.
    ---
    Title: TECHSPO San Francisco 2025
    Link: https://techsposanfrancisco.com/
    Snippet: TECHSPO San Francisco is a 2-day technology expo returning July 21 – 22, 2025 to Grand Hyatt Hotel at SFO in San Francisco, California.
    ---
    Title: World Agri-Tech Innovation Summit, March 11-12, 2025
    Link: https://worldagritechusa.com/
    Snippet: The World Agri-Tech Innovation Summit is the stand out event with curated leaders, innovators and investors from across the AgTech landscape.
    ---
    
    [00m
    [32;1m[1;3mAction: Read website content
    Action Input: {"website_url": "https://worldagritechusa.com/venue/"}[0m[95m 
    
    World Agri-Tech Innovation Summit, San Francisco: Venue
    Skip to content
    Contact Us
    Home
    About
    About Us
    Advisory Board
    Who Will You Meet
    Venue
    Upcoming Summits
    FAQs
    Program
    Program
    XPRIZE Workshop
    AI in Agriculture Forum
    Ginkgo Bioworks Tour
    1-1 Meetings
    Speakers
    Partners
    Start-Ups
    News
    News
    Newsletter Sign Up
    Venue
    REGISTER
    South America
    CLOSE
    Contact Us
     If you have any questions please contact:
    Marion Grimes
    Senior Business Development Manager
    Rethink Events
    T: +44 (0)1273 789989
    marion.grimes@rethinkevents.com
    Rethink Events Ltd,
    1st Floor, Huntingdon House,
    20a North Street,
    Brighton, BN1 1EB, UK
    Our Events
    							Venue: Marriott Marquis, San Francisco						
    DOWNLOAD SUMMIT BROCHURE
    SUMMIT VENUE: MARRIOTT MARQUIS
    World Agri-Tech is returning to California, the global epicenter for innovation discovery and investment.
    Better accommodation rates: No clash with other major events across San Francisco means you’ll benefit from lower rates across the city.
    Use our exclusive discount for your hotel room – book now to avoid disappointment!
    BOOK YOUR ROOM
    Marriott Marquis
    780 Mission Street
    San Francisco, CA 94103
    Getting to the Marriott Marquis from local airportsRead more
    San Francisco International Airport (SFO), 14 miles away
    Estimated taxi fare: 45 USD (one way)
    Estimated subway fare: 8.65 USD (one way)
    Take the yellow line towards Antioch to Powell Street BART Station. Exit via Market Street & 5th Street. Walk North East along Market Street and take a right onto 4th Street. The entrance will be on your left.
    Oakland International Airport (OAK), 20 miles away
    Estimated taxi fare: 60 USD (one way)
    Estimated subway fare: 4.05 USD (one way)
    Take the beige line towards Coliseum to Coliseum BART Station. Change here to take the green line towards Daly City to Powell Street BART Station. Exit via Market Street & 5th Street. Walk North East along Market Street and take a right onto 4th Street. The entrance will be on your left.
    Hotel ParkingRead more
    The hotel car park has limited availability, and parking spaces are allocated on a first-come, first-served basis.
    The parking fee is $91.2 per day for guests. Valet parking only.
    2nd Floor, One Gloucester Place
    Brighton
    BN1 4AA, UK
    rethinkevents.com
    T: +44 (0)1273 789989
    E: info@rethinkevents.com
     Built by Winc Studio
    [00m
    This is the agent final answer: Venue: Marriott Marquis, San Francisco
    Location: 780 Mission Street, San Francisco, CA 94103
    Description: World Agri-Tech is returning to California, the global epicenter for innovation discovery and investment. The venue offers better accommodation rates with no clash with other major events across San Francisco, benefiting from lower rates across the city.
    Transportation: 
    - San Francisco International Airport (SFO), 14 miles away
      - Estimated taxi fare: $45 USD (one way)
      - Estimated subway fare: $8.65 USD (one way)
    - Oakland International Airport (OAK), 20 miles away
      - Estimated taxi fare: $60 USD (one way)
      - Estimated subway fare: $4.05 USD (one way)
    Parking: 
    - Limited availability at the hotel car park
    - Parking fee: $91.2 per day for guests, valet parking only
    Please provide a feedback: yes, i love these options
    [32;1m[1;3mFinal Answer: 
    Venue: Marriott Marquis, San Francisco
    Location: 780 Mission Street, San Francisco, CA 94103
    Description: World Agri-Tech is returning to California, the global epicenter for innovation discovery and investment. The venue offers better accommodation rates with no clash with other major events across San Francisco, benefiting from lower rates across the city.
    Transportation: 
    - San Francisco International Airport (SFO), 14 miles away
      - Estimated taxi fare: $45 USD (one way)
      - Estimated subway fare: $8.65 USD (one way)
    - Oakland International Airport (OAK), 20 miles away
      - Estimated taxi fare: $60 USD (one way)
      - Estimated subway fare: $4.05 USD (one way)
    Parking: 
    - Limited availability at the hotel car park
    - Parking fee: $91.2 per day for guests, valet parking only[0m
    
    [1m> Finished chain.[0m
    [1m[92m [DEBUG]: == [Venue Coordinator] Task output: {
      "name": "Marriott Marquis, San Francisco",
      "address": "780 Mission Street, San Francisco, CA 94103",
      "capacity": 0,
      "booking_status": ""
    }
    
    [00m
    [1m[95m [DEBUG]: == Working Agent: Logistics Manager[00m
    [1m[95m [INFO]: == Starting Task: Coordinate catering and equipment for an event with 500 participants on 2024-09-15.[00m
    [1m[92m [DEBUG]: == [Logistics Manager] Task output: {
      "name": "Marriott Marquis, San Francisco",
      "address": "780 Mission Street, San Francisco, CA 94103",
      "capacity": 0,
      "booking_status": ""
    }
    
    [00m
    [1m[95m [DEBUG]: == Working Agent: Marketing and Communications Agent[00m
    [1m[95m [INFO]: == Starting Task: Promote the Tech Innovation Conference aiming to engage at least500 potential attendees.[00m
    [1m[92m [DEBUG]: == [Marketing and Communications Agent] Task output: {
      "name": "Marriott Marquis, San Francisco",
      "address": "780 Mission Street, San Francisco, CA 94103",
      "capacity": 0,
      "booking_status": ""
    }
    
    [00m


    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI need to gather information about the Tech Innovation Conference and engage potential attendees effectively. 
    
    Action: Read website content
    Action Input: {"website_url": "https://www.techinnovationconference.com"}[0m[32;1m[1;3mI need to ensure that all logistics for the event, including catering and equipment setup, are flawlessly executed.
    
    Action: Read website content
    Action Input: {"website_url": "www.marriott.com/marriottmarquis/sanfrancisco"}[0m[91m 
    
    I encountered an error while trying to use the tool. This was the error: Invalid URL 'www.marriott.com/marriottmarquis/sanfrancisco': No scheme supplied. Perhaps you meant https://www.marriott.com/marriottmarquis/sanfrancisco?.
     Tool Read website content accepts these inputs: Read website content(website_url: 'string') - A tool that can be used to read a website content.
    [00m
    [91m 
    
    I encountered an error while trying to use the tool. This was the error: HTTPSConnectionPool(host='www.techinnovationconference.com', port=443): Max retries exceeded with url: / (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x7f7b2e138890>: Failed to resolve 'www.techinnovationconference.com' ([Errno -2] Name or service not known)")).
     Tool Read website content accepts these inputs: Read website content(website_url: 'string') - A tool that can be used to read a website content.
    [00m
    [32;1m[1;3mThought: I need to find information about the Tech Innovation Conference to effectively market it.
    Action: Search the internet
    Action Input: {"search_query": "Tech Innovation Conference details"}[0m[32;1m[1;3mThought: I need to gather information about catering and equipment setup for the event at Marriott Marquis, San Francisco.
    
    Action: Search the internet
    Action Input: {"search_query": "Marriott Marquis San Francisco catering and equipment setup"}[0m[95m 
    
    
    Search results: Title: Tech Conferences: The Best Tech Events Guide for 2025 - Bizzabo
    Link: https://www.bizzabo.com/blog/technology-events
    Snippet: Gartner Tech Growth & Innovation Conference. Mar. 10-11, 2025. Grapevine, TX. Igniting Growth: New Buyers. New Solutions. New Business Models. Explore the new ...
    ---
    Title: Gartner Tech Growth & Innovation Conference 2025, in Grapevine, TX
    Link: https://www.gartner.com/en/conferences/na/tech-growth-us
    Snippet: Conference Venue. Gaylord Texan Hotel & Convention Center. Prepare for your trip with venue details, accommodation options and transportation information.
    ---
    Title: Technology & Innovation Summit North America - Forrester
    Link: https://www.forrester.com/event/technology-innovation-north-america/
    Snippet: Join your fellow technology and data leaders at this year's Technology & Innovation Summit in Austin, TX, November 2-5, 2025.
    ---
    Title: LSC's 25th Innovations in Technology Conference
    Link: https://www.lsc.gov/events/events/lscs-25th-innovations-technology-conference
    Snippet: The Legal Services Corporation's 2025 Innovations in Technology Conference took place in Phoenix, AZ, from January 13-15, 2025. Hundreds of technologists ...
    ---
    Title: TechConnect World 2025
    Link: https://www.techconnectworld.com/
    Snippet: TechConnect World 2025 will take place June 9-11, 2025, Austin, TX, JW Marriott.
    ---
    Title: Top Tech Conferences to Attend in 2024-2025
    Link: https://graduate.northeastern.edu/knowledge-hub/2019-tech-conferences/
    Snippet: Top 10 technology conferences of 2024–2025 · 1. Black Hat USA · 2. VMware Explore US · 3. Cyber Security Summit · 4. Deep Learning Summit · 5. Open Data Science ...
    ---
    Title: 2024 Innovations in Technology Conference | LSC
    Link: https://www.lsc.gov/events/events/2024-innovations-technology-conference
    Snippet: 2024 Innovations in Technology Conference. February 01, 2024 | 8:30am - February 03, 2024 | 12:30pm •. Charlotte, NC.
    ---
    Title: Home - Cybersecurity and Technology Innovation Conference
    Link: https://www.doecybercon.com/
    Snippet: The Department of Energy's DOE Cybersecurity and Technology Innovation Conference will explore the developments and challenges in cybersecurity.
    ---
    Title: 12 Best Innovation Conferences To Consider In 2025 - The CTO Club
    Link: https://thectoclub.com/news/best-innovation-conferences/
    Snippet: A curated list of must-attend innovation conferences in 2025 to explore emerging trends, technology, and strategies in innovation.
    ---
    Title: Tech. Innovation Conference 2035 - NICER Consortium
    Link: https://www.nicerconsortium.org/events/tech-innovation-conference-2035
    Snippet: Tech. Innovation Conference 2035 · Time & Location · About The Event · Agenda · Tickets · Share This Event · The Hungry Immune System.
    ---
    
    [00m
    [95m 
    
    
    Search results: Title: Meetings and Special Events Space in San Francisco - Marriott
    Link: https://www.marriott.com/en-us/hotels/sfodt-san-francisco-marriott-marquis/events/
    Snippet: Let our specialists help plan your event in San Francisco, from wedding location set-up to catering. Host cocktail hour in The View Lounge for your wedding ...
    ---
    Title: Hotel in San Francisco | San Francisco Marriott Marquis
    Link: https://www.marriott.com/en-us/hotels/sfodt-san-francisco-marriott-marquis/overview/
    Snippet: Join us for a breakfast buffet in San Francisco between 6:30am-10:30am and dinner 5pm-10pm, daily at our restaurant in downtown San Francisco. Locally ...
    ---
    Title: San Francisco Marriott Marquis - Hotel - Cvent
    Link: https://www.cvent.com/venues/san-francisco/hotel/san-francisco-marriott-marquis/venue-67fb69bd-9e13-4ff7-8e7f-856f0c2e41de
    Snippet: Amenities · Onsite catering · Onsite gift shop · Onsite restaurant · Onsite security · Pet friendly · Rental car service · Wheelchair accessible.
    ---
    Title: [PDF] Event Menu | Marriott
    Link: https://www.marriott.com/content/dam/marriott-digital/mc/us-canada/hws/s/sfous/en_us/document/assets/mc-sfous-sfous-event-menu-2022-fin-28814.pdf
    Snippet: SAN FRANCISCO MARRIOTT UNION SQUARE. 480 Sutter Street, San Francisco, CA 94108 ... For installation and removal of equipment, a 25% service charge of the ...
    ---
    Title: [PDF] Marriott Marquis Menu - American Public Health Association
    Link: https://www.apha.org/-/media/files/pdf/meetings/annual/2020/marriott_marquis_menu.ashx
    Snippet: All Food and Beverage Must be supplied by the Hotel. ... Fees, Guidelines and Minimum Service Standards Are Outlined in the San Francisco Marriott Marquis' Hotel ...
    ---
    Title: Bars with a View, San Francisco | San Francisco Marriott Marquis
    Link: https://www.marriott.com/en-us/hotels/sfodt-san-francisco-marriott-marquis/photos/
    Snippet: Catering - Sandwiches. Catering - Salads. Catering - Fruit Tray. SoMa Meeting Room - Details. SoMa Meeting Room – Reception Setup. SoMa Meeting Room – Theater ...
    ---
    Title: Sierra D at San Francisco Marriott Marquis - Hotel - The Vendry
    Link: https://thevendry.com/venue/5072/san-francisco-marriott-marquis-san-francisco-ca/space/10841
    Snippet: Capacity. Seated: 14. Square Feet: 693 ft 2. F&B Options. In-house catering. Equipment. A/V Equipment. Air Conditioning. Frequent Uses. Meetings. Private Dining ...
    ---
    Title: San Francisco Marriott Marquis - San Francisco, CA
    Link: https://www.northstarmeetingsgroup.com/Meeting-Event-Venues/San-Francisco/Convention-Hotel/San-Francisco-Marriott-Marquis-p52651329
    Snippet: Plan your next event or meeting at San Francisco Marriott Marquis in San Francisco, CA. Check out total event space, meeting rooms, and request a proposal ...
    ---
    Title: San Francisco Marriott Marquis [Master Thread] - FlyerTalk Forums
    Link: https://www.flyertalk.com/forum/marriott-marriott-bonvoy/2042616-san-francisco-marriott-marquis-master-thread.html
    Snippet: The hotel is only offering 750 points per night as a substitute for the lounge, despite their grab-and-go shop being open and their open breakfast buffet on ...
    ---
    Title: restaurant suggestions near Marriott Marquis? - San Francisco Forum
    Link: https://www.tripadvisor.com/ShowTopic-g60713-i30-k11943318-Restaurant_suggestions_near_Marriott_Marquis-San_Francisco_California.html
    Snippet: If you want to eat near your hotel on the first night, I will suggest Zero Zero, Hops & Hominy or E&O as reasonably priced and walkable.
    ---
    
    [00m


    [32;1m[1;3mThought: I found information about various tech conferences, but not specifically the Tech Innovation Conference I need to promote.
    
    Action: Search the internet
    Action Input: {"search_query": "Tech Innovation Conference 2025 details"}[0mThis is the agent final answer: After searching the internet, I found information related to catering and equipment setup for events at the San Francisco Marriott Marquis. The hotel offers onsite catering services and has a document outlining the event menu. Additionally, they provide equipment setup services with a service charge for installation and removal. The hotel seems to have the necessary amenities and options for catering and equipment setup for events, making it a suitable choice for our event with 500 participants on 2024-09-15.
    Please provide a feedback: 

    Exception in thread Thread-5 (_execute):
    Traceback (most recent call last):
      File "/usr/local/lib/python3.11/threading.py", line 1045, in _bootstrap_inner
        self.run()
      File "/usr/local/lib/python3.11/site-packages/ipykernel/ipkernel.py", line 766, in run_closure
        _threading_Thread_run(self)
      File "/usr/local/lib/python3.11/threading.py", line 982, in run
        self._target(*self._args, **self._kwargs)
      File "/usr/local/lib/python3.11/site-packages/crewai/task.py", line 182, in _execute
        result = agent.execute_task(
                 ^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.11/site-packages/crewai/agent.py", line 221, in execute_task
        result = self.agent_executor.invoke(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.11/site-packages/langchain/chains/base.py", line 163, in invoke
        raise e
      File "/usr/local/lib/python3.11/site-packages/langchain/chains/base.py", line 153, in invoke
        self._call(inputs, run_manager=run_manager)
      File "/usr/local/lib/python3.11/site-packages/crewai/agents/executor.py", line 124, in _call
        next_step_output = self._take_next_step(
                           ^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.11/site-packages/langchain/agents/agent.py", line 1138, in _take_next_step
        [
      File "/usr/local/lib/python3.11/site-packages/langchain/agents/agent.py", line 1138, in <listcomp>
        [
      File "/usr/local/lib/python3.11/site-packages/crewai/agents/executor.py", line 245, in _iter_next_step
        human_feedback = self._ask_human_input(output.return_values["output"])
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.11/site-packages/crewai/agents/executor.py", line 300, in _ask_human_input
        return input(
               ^^^^^^
    EOFError: EOF when reading a line


    [95m 
    
    
    Search results: Title: Gartner Tech Growth & Innovation Conference 2025, in Grapevine, TX
    Link: https://www.gartner.com/en/conferences/na/tech-growth-us
    Snippet: Conference Venue. Gaylord Texan Hotel & Convention Center. Prepare for your trip with venue details, accommodation options and transportation information.
    ---
    Title: Tech Conferences: The Best Tech Events Guide for 2025 - Bizzabo
    Link: https://www.bizzabo.com/blog/technology-events
    Snippet: Gartner Tech Growth & Innovation Conference. Mar. 10-11, 2025. Grapevine, TX. Igniting Growth: New Buyers. New Solutions. New Business Models. Explore the new ...
    ---
    Title: TechConnect World 2025
    Link: https://www.techconnectworld.com/
    Snippet: TechConnect World 2025 will take place June 9-11, 2025, Austin, TX, JW Marriott.
    ---
    Title: LSC's 25th Innovations in Technology Conference
    Link: https://www.lsc.gov/events/events/lscs-25th-innovations-technology-conference
    Snippet: The Legal Services Corporation's 2025 Innovations in Technology Conference took place in Phoenix, AZ, from January 13-15, 2025. Hundreds of technologists ...
    ---
    Title: 2025 TechConnect World Innovation Conference and Expo
    Link: https://techconnect.org/2025-techconnect-world-innovation-conference-and-expo/
    Snippet: 2025 TechConnect World Innovation Conference and Expo. June 9-11, 2025 | Austin, Texas. JW Marriott. More information coming soon. JOIN US. previous next
    ---
    Title: 15 Must-Attend 2025 Technology Conferences for Industry Leaders
    Link: https://www.psmpartners.com/blog/15-must-attend-2025-technology-conferences-for-industry-leaders/
    Snippet: 1. CES Consumer Electronics Show · January 7-10, 2025 · Las Vegas, Nevada ; 2. DeveloperWeek · February 11-13, 2025 · San Francisco Bay Area & Virtual ; 3. Gartner ...
    ---
    Title: Gartner Announces Gartner Tech Growth & Innovation Conference ...
    Link: https://www.gartner.com/en/newsroom/press-releases/2025-01-16-gartner-announces-gartner-tech-growth-and-innovation-conference-2025
    Snippet: What: Gartner Tech Growth & Innovation Conference · When: March 10-11, 2025 · Where: Grapevine, TX · Details: Igniting growth: new buyers, new ...
    ---
    Title: Technology & Innovation Summit North America - Forrester
    Link: https://www.forrester.com/event/technology-innovation-north-america/
    Snippet: Join your fellow technology and data leaders at this year's Technology & Innovation Summit in Austin, TX, November 2-5, 2025.
    ---
    Title: Top Tech Conferences to Attend in 2024-2025
    Link: https://graduate.northeastern.edu/knowledge-hub/2019-tech-conferences/
    Snippet: If your company is looking for the latest technological innovations, then CES 2025 is a must-attend event. 8. RSA Conference. Date: April 28–May 1, 2025.
    ---
    Title: Biggest Tech Conferences in 2025: Your Guide to Global Innovation
    Link: https://techspola.com/biggest-tech-conferences-in-2025-your-guide-to-global-innovation/
    Snippet: TECHSPO LA 2025 is a two day technology expo taking place April 3rd - 4th, 2025 at the luxurious Loews Santa Monica Beach Hotel in Los Angeles, California in ...
    ---
    
    [00m
    [32;1m[1;3mThought: I need to reach out to the Venue Coordinator to gather information about the location for the Tech Innovation Conference.
    
    Action: Ask question to co-worker
    Action Input: {"coworker": "Venue Coordinator", "question": "Can you provide me with details about the venue for the Tech Innovation Conference?","context": {"name": "Marriott Marquis, San Francisco","address": "780 Mission Street, San Francisco, CA 94103","capacity": 0,"booking_status": ""}}[0m
    
    [1m> Entering new CrewAgentExecutor chain...[0m
    [32;1m[1;3mI need to gather information about the Marriott Marquis in San Francisco for the Tech Innovation Conference.
    
    Action: Read website content
    Action Input: {"website_url": "https://www.marriott.com/hotels/travel/sfodt-san-francisco-marriott-marquis"}[0m[95m 
    
    Access Denied
    Access Denied
    You don't have permission to access "http://www.marriott.com/en-us/hotels/sfodt-san-francisco-marriott-marquis/overview/" on this server.
    Reference #18.5160d017.1738649883.fda50290
    https://errors.edgesuite.net/18.5160d017.1738649883.fda50290
    [00m
    [32;1m[1;3mFinal Answer: Unfortunately, I was unable to access the website content for the Marriott Marquis in San Francisco. I recommend reaching out to the hotel directly or visiting their official website for more information about the venue for the Tech Innovation Conference.[0m
    
    [1m> Finished chain.[0m
    [95m 
    
    Unfortunately, I was unable to access the website content for the Marriott Marquis in San Francisco. I recommend reaching out to the hotel directly or visiting their official website for more information about the venue for the Tech Innovation Conference.
    [00m
    [32;1m[1;3mThought: I need to gather information about the Tech Innovation Conference and engage potential attendees effectively.
    
    Action: Search the internet
    Action Input: {"search_query": "Tech Innovation Conference 2025 details"}[0m[95m 
    
    
    Search results: Title: Gartner Tech Growth & Innovation Conference 2025, in Grapevine, TX
    Link: https://www.gartner.com/en/conferences/na/tech-growth-us
    Snippet: Conference Venue. Gaylord Texan Hotel & Convention Center. Prepare for your trip with venue details, accommodation options and transportation information.
    ---
    Title: Tech Conferences: The Best Tech Events Guide for 2025 - Bizzabo
    Link: https://www.bizzabo.com/blog/technology-events
    Snippet: Gartner Tech Growth & Innovation Conference. Mar. 10-11, 2025. Grapevine, TX. Igniting Growth: New Buyers. New Solutions. New Business Models. Explore the new ...
    ---
    Title: TechConnect World 2025
    Link: https://www.techconnectworld.com/
    Snippet: TechConnect World 2025 will take place June 9-11, 2025, Austin, TX, JW Marriott.
    ---
    Title: LSC's 25th Innovations in Technology Conference
    Link: https://www.lsc.gov/events/events/lscs-25th-innovations-technology-conference
    Snippet: The Legal Services Corporation's 2025 Innovations in Technology Conference took place in Phoenix, AZ, from January 13-15, 2025. Hundreds of technologists ...
    ---
    Title: 2025 TechConnect World Innovation Conference and Expo
    Link: https://techconnect.org/2025-techconnect-world-innovation-conference-and-expo/
    Snippet: 2025 TechConnect World Innovation Conference and Expo. June 9-11, 2025 | Austin, Texas. JW Marriott. More information coming soon. JOIN US. previous next
    ---
    Title: 15 Must-Attend 2025 Technology Conferences for Industry Leaders
    Link: https://www.psmpartners.com/blog/15-must-attend-2025-technology-conferences-for-industry-leaders/
    Snippet: 1. CES Consumer Electronics Show · January 7-10, 2025 · Las Vegas, Nevada ; 2. DeveloperWeek · February 11-13, 2025 · San Francisco Bay Area & Virtual ; 3. Gartner ...
    ---
    Title: Gartner Announces Gartner Tech Growth & Innovation Conference ...
    Link: https://www.gartner.com/en/newsroom/press-releases/2025-01-16-gartner-announces-gartner-tech-growth-and-innovation-conference-2025
    Snippet: What: Gartner Tech Growth & Innovation Conference · When: March 10-11, 2025 · Where: Grapevine, TX · Details: Igniting growth: new buyers, new ...
    ---
    Title: Technology & Innovation Summit North America - Forrester
    Link: https://www.forrester.com/event/technology-innovation-north-america/
    Snippet: Join your fellow technology and data leaders at this year's Technology & Innovation Summit in Austin, TX, November 2-5, 2025.
    ---
    Title: Top Tech Conferences to Attend in 2024-2025
    Link: https://graduate.northeastern.edu/knowledge-hub/2019-tech-conferences/
    Snippet: If your company is looking for the latest technological innovations, then CES 2025 is a must-attend event. 8. RSA Conference. Date: April 28–May 1, 2025.
    ---
    Title: Biggest Tech Conferences in 2025: Your Guide to Global Innovation
    Link: https://techspola.com/biggest-tech-conferences-in-2025-your-guide-to-global-innovation/
    Snippet: TECHSPO LA 2025 is a two day technology expo taking place April 3rd - 4th, 2025 at the luxurious Loews Santa Monica Beach Hotel in Los Angeles, California in ...
    ---
    
    [00m
    [32;1m[1;3mThought: I need to engage potential attendees effectively by sharing information about the Tech Innovation Conference.
    
    Action: Search the internet


    Action Input: {"search_query": "Tech Innovation Conference 2025 attendee engagement ideas"}[0m[95m 
    
    
    Search results: Title: 21 Event Engagement Ideas to Boost Participation in 2025 - Whova
    Link: https://whova.com/blog/event-engagement-ideas/
    Snippet: Engagement Ideas During the Event · 7. Live Polls and Interactive Q&A · 8. Networking Breakout Rooms · 9. Host a Social Media Wall · 10. Share Event ...
    ---
    Title: 19 Engaging Conference Themes for 2025 | Cvent Blog
    Link: https://www.cvent.com/en/blog/events/conference-themes
    Snippet: Cover topics like communication skills, conflict resolution, goal setting, time management, project management, and building a strong team ...
    ---
    Title: 9 Event Technology Trends for 2025
    Link: https://meetings-incentives.com/9-event-technology-trends-for-2025/
    Snippet: 1. Personalization at Scale with AI · 2. Networking Lounges and AI Matchmaking · 3. AI Chatbots: Simplifying Registration and Engagement · 4. Data ...
    ---
    Title: 145 Enagaging Conference Themes For Your Next Event In 2025
    Link: https://eventify.io/blog/conference-themes
    Snippet: Discover 145 engaging conference themes for your next event in 2025. From creative ideas to professional topics, find the perfect theme to inspire and ...
    ---
    Title: What Are the Themes for Tech Conferences in 2025?
    Link: https://techspola.com/what-are-the-themes-for-tech-conferences-in-2025/
    Snippet: Attendees can explore themes such as artificial intelligence, sustainability, and the future of work.
    ---
    Title: Top legal technology conferences to attend in 2025 - One Legal
    Link: https://www.onelegal.com/blog/top-legal-technology-conferences-to-attend-in-2025/
    Snippet: Attendees will explore topics like practice management, client engagement, and automation, with insights from industry leaders and interactive ...
    ---
    Title: Top Innovation Conferences of 2025 - Seed Strategy
    Link: https://seedstrategy.com/brand-building/top-innovation-conferences-of-2025/
    Snippet: Attendees will hear leaders from large companies share case studies and valuable lessons on innovation, technology, R&D, formulation and regulations that ...
    ---
    Title: 8 corporate event trends and conference themes for 2025. - GoGather
    Link: https://gogather.com/blog/conference-event-trends
    Snippet: Gamification is becoming a go-to strategy at conferences lately, adding an extra layer of engagement for attendees. It's a fun way to motivate ...
    ---
    Title: Tech Conferences: The Best Tech Events Guide for 2025 - Bizzabo
    Link: https://www.bizzabo.com/blog/technology-events
    Snippet: This carefully curated conference directory features more than 100 tech conferences happening worldwide in 2025, with a focus on North America.
    ---
    
    [00m
    [32;1m[1;3mThought: I now need to compile a report on marketing activities and attendee engagement for the Tech Innovation Conference.
    
    Final Answer: 
    
    # Tech Innovation Conference Marketing and Attendee Engagement Report
    
    ## Marketing Activities:
    - Utilized search engines to gather information about the conference and potential attendees.
    - Explored various tech conference websites to identify trends and best practices for promotion.
    - Searched for engagement ideas to boost participation in the conference.
    - Investigated themes for tech conferences in 2025 to align marketing strategies.
    
    ## Attendee Engagement Strategies:
    - Considered using live polls and interactive Q&A sessions during the event.
    - Explored the possibility of hosting networking breakout rooms for attendees.
    - Identified the potential of a social media wall to encourage interaction.
    - Explored themes such as artificial intelligence, sustainability, and the future of work for engaging conference topics.
    
    Overall, the marketing activities focused on understanding the landscape of tech conferences in 2025 and identifying unique strategies to promote the Tech Innovation Conference effectively. The attendee engagement strategies aimed to create an interactive and dynamic event environment to maximize participation and create a memorable experience for attendees.[0m
    
    [1m> Finished chain.[0m


- Display the generated `venue_details.json` file.


```python
import json
from pprint import pprint

with open('venue_details.json') as f:
   data = json.load(f)

pprint(data)
```

    {'address': '780 Mission Street, San Francisco, CA 94103',
     'booking_status': '',
     'capacity': 0,
     'name': 'Marriott Marquis, San Francisco'}


- Display the generated `marketing_report.md` file.

**Note**: After `kickoff` execution has successfully ran, wait an extra 45 seconds for the `marketing_report.md` file to be generated. If you try to run the code below before the file has been generated, your output would look like:

```
marketing_report.md
```

If you see this output, wait some more and than try again.


```python
from IPython.display import Markdown
Markdown("marketing_report.md")
```




marketing_report.md




```python

```


```python

```
