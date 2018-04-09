# Vizualization inspired from here http://ramiro.org/notebook/sherlock-holmes-canon-wordcloud/
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from scipy.misc import imread
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from wordcloud import WordCloud, STOPWORDS

mpl.style.use('classic')
limit = 1000
infosize = 12

title = 'Most frequent words in the speech of Adolf Hitler'
chartinfo = 'Author: Victor Vrabie'
footer = 'The {} most frequent words used in the Adolf Hitler speech at 4th anniverssay of comming to power .\n{}'.format(limit, chartinfo)
font = '/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf'
fontcolor='#fafafa'
bgcolor = '#000000'
english_stopwords = set(stopwords.words('english')) | STOPWORDS | ENGLISH_STOP_WORDS


def grey_color(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'hsl(0, 0%%, %d%%)' % random.randint(50, 100)

wordcloud = WordCloud(
    max_words=limit,
    stopwords=english_stopwords,
    mask=imread(r'C:\Users\VictorVrabie\Downloads\stalin.png'),
    background_color=bgcolor
).generate(data_stemmed)

fig = plt.figure()
fig.set_figwidth(21)
fig.set_figheight(27)

plt.imshow(wordcloud.recolor(color_func=grey_color, random_state=3))
plt.title(title, color=fontcolor, size=30, y=1.01)
plt.annotate(footer, xy=(0, -.025), xycoords='axes fraction', fontsize=infosize, color=fontcolor)
plt.axis('off')
plt.show()

plt.savefig('plot.png')