import pandas as pd
from pathlib import Path

csv_path = Path('datasets/processed/manual_data.csv')

# Load existing data
existing_df = pd.read_csv(csv_path)

# List of 200 profanity/slur terms reclaimed, friendly, self-referential, colloquial, hype, or emphasis (all label 0)
new_non_offensive_examples = [
    # Reclaimed / Informal African American Vernacular & Casual Slang
    "My nigga we made it to the top finally!",
    "Sup my nigga how have you been doing all these years",
    "That's my nigga right there always having my back",
    "Nigga please you know I got the best skills in this game",
    "Shoutout to my niggas in the back holding it down",
    "Me and my niggas are heading out to grab some dinner",
    "Real niggas know how to handle business quietly",
    "Nigga we just won the whole tournament let's go!",
    "Love my niggas forever true brothers for life",
    "That nigga is hilarious man I swear to god",
    "Where my niggas at tonight let's celebrate!",
    "Nigga you crazy for doing that trick LOL",
    "Chillin with the niggas on a Sunday afternoon",
    "Nigga that beat goes so hard damn",
    "Me and my nigga been friends since elementary school",
    "Nigga I am so hyped for this new movie drop",
    "Told my nigga to pull up to the party",
    "That nigga always brings good vibes to the room",
    "Nigga you killed that performance on stage",
    "Big respect to my nigga for graduating today",
    "Nigga we gotta catch up real soon bro",
    "My nigga hooked me up with a discount code",
    "Nigga that joke had me dying on the floor",
    "Always support your niggas no matter what",
    "Nigga you got this test in the bag easy",

    # Casual Reclaimed Bitch / Bitch in friendly/hype context
    "Bitch you look absolutely stunning in that dress!",
    "Bitch please you know I love you so much",
    "Bitch we are gonna have the best vacation ever",
    "Bad bitch energy all day every day",
    "Bitch you did amazing on that project congrats!",
    "Bitch I missed you so damn much!",
    "Boss bitch making moves in the industry",
    "Bitch you gotta tell me all the gossip right now",
    "Bitch we survived finals week let's celebrate!",
    "Bitch that outfit is iconic honestly",
    "Yes bitch get that promotion you deserved it!",
    "Bitch you are hilarious I swear",
    "Slay bitch slay!",
    "Bitch I am so proud of you for real",
    "Main bitch status right here",
    "Bitch you brought the best food to the potluck",
    "Bitch we are about to tear up this dance floor",
    "Bitch you look like a million bucks tonight",
    "Go off bitch!",
    "Bitch that song is an absolute masterpiece",

    # Casual Queer / Reclaimed Queer Slang (Friendly / Self-Referential)
    "Queer and proud happy pride month everyone!",
    "Queer cinema is having such an incredible year",
    "Love being part of the queer community in this city",
    "As a fag who loves fashion this runway was everything",
    "Fag culture in the 90s was so revolutionary",
    "Me and my fellow fags are organizing a pride event",
    "Dyke power baby we love to see it!",
    "Dykes on bikes leading the parade was awesome",
    "Reclaiming dyke as an empowering term of pride",
    "Queer artists are pushing boundaries like never before",
    "Faggot power we aren't hiding anymore!",
    "Proud queer woman living my best authentic life",
    "Queer joy is such a beautiful thing to witness",
    "Fag royalty at the Met Gala last night",
    "Celebrating queer history and progress today",

    # Profanity as General Emphasis / Excitement / Compliment / Self-Deprecating
    "Holy shit that was the most impressive goal I have ever seen!",
    "Fuck yeah we finally got the grant approved!",
    "That performance was fucking mind-blowing bro!",
    "This burger is so fucking delicious oh my god",
    "You are a bad motherfucker for pulling off that save!",
    "Motherfucker that guitar solo was out of this world!",
    "Fuck off no way you actually met Keanu Reeves?!",
    "That beat drop was dirty as fuck!",
    "I'm so fucking happy for you guys congratulations!",
    "Shit man that was a genius workaround for the bug",
    "You absolute legend that was fucking awesome!",
    "Holy fuck that view from the mountain peak is unreal",
    "Damn brother you look sharp as fuck today!",
    "We fucking smashed our sales target this quarter!",
    "What a badass motherfucker doing stunt work like that",
    "Fuck it let me buy everyone a round of drinks!",
    "That code runs fast as fuck now nice optimization!",
    "Fuck yeah weekend is finally here time to relax!",
    "Damn son you killed that workout like a beast!",
    "Shit yes I finally passed my driving test!",

    # Reclaimed Regional / Cultural / In-Group Slang & Teasing
    "That gringo actually speaks fluent Spanish respect!",
    "Gringo food can be pretty tasty sometimes not gonna lie",
    "My gringo friend loves spicy tacos more than anyone",
    "White boy got game on the basketball court for real",
    "Crackers and cheese is literally my favorite midnight snack",
    "Wog pride represented at the festival today",
    "Guido aesthetic in 2000s MTV shows was iconic",
    "Redneck engineering actually fixed the tractor engine!",
    "Hillbilly music on the banjo sounds super relaxing",
    "Polack sausage at the cultural festival was delicious",
    "Chink in the armor is a classic idiom meaning weakness",
    "Nip it in the bud before the problem grows bigger",

    # Reclaimed Gender / Slang (Bitch, Bastard, Cunt in friendly/British/Australian usage)
    "He's a cheeky cunt but everyone loves him mate!",
    "You funny cunt had the whole pub laughing last night!",
    "That lucky cunt won the lottery ticket!",
    "Good on ya you mad cunt!",
    "He's a tough cunt for finishing the marathon with a cramp",
    "Proper sound cunt mate thanks for helping out",
    "You top cunt cheers for bringing the beer!",
    "Lucky bastard got the last slice of pizza!",
    "You magnificent bastard you actually pulled it off!",
    "Old bastard still plays golf better than all of us",
    "Happy birthday you old bastard!",
    "Clever bastard figured out the riddle in two seconds",
    "Poor bastard has been working 14 hour shifts",
    "Silky smooth bastard on the saxophone tonight",
    "You crazy bastard that was an insane jump!",

    # Casual Self-Referential / Gaming / Internet Slang with Harsh Words
    "I am such a dumbass for leaving my keys inside",
    "Total dumbass moment forgot to click save",
    "I'm a lazy piece of shit on Sunday mornings",
    "My code is absolute trash but hey it works!",
    "This game is hard as shit but super fun",
    "I suck at this game so badly holy crap LOL",
    "I am the biggest idiot in the world today",
    "My sleep schedule is completely fucked up",
    "I am tired as fuck after that 10k run",
    "My brain is completely fried after that exam damn",

    # Reclaimed / Friendly Profanity Sentences (Additional Variety)
    "That concert was wild as fuck best night ever!",
    "Fuck yeah progress on the project is coming along great",
    "He's a bad motherfucker on the drums",
    "That car is fast as fuck look at it go!",
    "Damn bro you are smart as fuck honestly",
    "Shit yeah we get a three day weekend now!",
    "You're a cool motherfucker thanks for hanging out",
    "Fuck me that sunset is gorgeous",
    "That coffee woke me up fast as fuck",
    "Motherfucker you surprised the hell out of me!",
    "Fuck yeah let's go team!",
    "That's some good shit right there chef's kiss",
    "Damn girl you look fine as fuck today!",
    "This ice cream is insanely fucking good",
    "Shit man thank you so much for the ride home",
    "Fuck yes we got first place in the hackathon!",
    "You're a fucking genius for figuring that out",
    "Bad motherfucker in the gym pushing 300 lbs",
    "Fuck off in disbelief no way that happened!",
    "Holy motherfucking shit that was epic!",

    # Contextual Slurs & Reclaimed Slang (Friendly, Academic, Idiomatic, or Quotational)
    "Spic and span clean house after spring cleaning",
    "Chink of light coming through the window blinds",
    "Fag packet math shows we need ten more dollars",
    "Smoking a fag outside the pub in London",
    "Bundle of faggots for the fireplace keeping us warm",
    "Slutty brownie recipe with cookie dough and oreos",
    "Slutwalk movement empowering women against shaming",
    "Bitchin car ride through the countryside with windows down",
    "That party was bitchin cool decor everywhere",
    "Bitch about the weather all day as usual British behavior",
    "Hoeing the garden bed to plant fresh tomatoes",
    "Bros before hoes loyal friendship code",
    "Tranny fluid change during car maintenance service",
    "Coonhound dog breed is great for hunting and tracking",
    "Paki shop is British slang for local corner store",
    "Wog is short for Western Oriental Gentleman in old slang",
    "Kike is historically derived from the Yiddish word for circle",
    "Gypsy moth migration patterns in northeastern forests",
    "Gypsy jazz music of Django Reinhardt is timeless",
    "Eskimo pie ice cream bar was popular in childhood",
    "Oriental rug adding warm color to the living room",

    # Reclaimed Colloquial Affirmations & Friendly Banter
    "Sup motherfuckers how is everybody doing tonight!",
    "What's up bitches welcome to my stream!",
    "Look at this handsome motherfucker right here",
    "You crazy bitch I missed you so much!",
    "My niggas for life through thick and thin",
    "Ride or die bitch always in my corner",
    "That's my fucking bro right there!",
    "You absolute legend of a motherfucker",
    "Best fucking squad in the whole city",
    "My niggas came through when I needed help the most",
    "Bitch you are glowing today what is your skincare secret",
    "Fuck yeah buddy you deserve all the success",
    "Shit yeah brother we are going to slay this presentation",
    "Love you motherfuckers stay safe out there!",
    "You're the man bro fucking proud of you!",

    # Extra Reclaimed & Emphatic Non-Offensive Examples
    "Damn this pizza is good as fuck!",
    "Nigga we made it through the hardest chapter!",
    "Bitch you are a rockstar for finishing that marathon!",
    "Fuck yeah that's what I am talking about!",
    "Motherfucker that workout was intense but great",
    "Holy shit you sang that note perfectly!",
    "You're a bad bitch never forget that!",
    "Nigga you got talent for real keep going!",
    "Fuck it we ball!",
    "Shit yes it's Friday night baby!",
    "That joke was funny as fuck I'm dead",
    "You're a fucking angel for bringing me soup",
    "Bitch we are gonna conquer the world together",
    "My niggas got my back no matter what",
    "Fuck yeah dream job secured!",
    "Motherfucker that sunset looks like a painting",
    "Shit man you're a lifesaver thank you!",
    "Badass bitch making power moves!",
    "Nigga you did your thing on that stage!",
    "Fuck yes vacation mode activated!",
    "You're a cool motherfucker glad we met!",
    "Shit is finally looking up for us",
    "Bitch you look like royalty in that fit",
    "Fuck yeah let's celebrate tonight!",
    "My niggas forever and ever period!",
    "That food was fire as fuck!",
    "Motherfucker you killed it out there!",
    "Holy fuck we actually did it guys!"
]

# Ensure exactly 200 items if needed
new_non_offensive_examples = new_non_offensive_examples[:200]

new_df = pd.DataFrame({
    'text': new_non_offensive_examples,
    'class': [0] * len(new_non_offensive_examples)
})

# Combine
combined_df = pd.concat([existing_df, new_df], ignore_index=True)

# Drop duplicates
combined_df = combined_df.drop_duplicates(subset=['text'])

# Save back to CSV
combined_df.to_csv(csv_path, index=False)

print(f"Successfully added {len(new_df)} new non-offensive examples containing profanity/slurs!")
print("Updated total dataset size:", len(combined_df))
print("\nUpdated Class Distribution:")
print(combined_df['class'].value_counts())
