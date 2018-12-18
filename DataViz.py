import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import numpy as np


def wordmap(df, i=1):
    # Seting good ratings and bad ratings
    dfwc = df.filter(['reviewText', 'overall'], axis=1)
    dfwc['GoodRating'] = np.where(dfwc.overall > 4, 1, 0)
    df_pos = dfwc[dfwc['GoodRating'] == i]

    # Concatening positive reviews
    reviewpos = []
    for review in df_pos['reviewText'].items():
        reviewpos.append(review[1])
    strpos = ''.join(reviewpos)

    # Generate a word cloud image for positive reviews
    wordcloud = WordCloud(
        max_font_size=40, background_color='white').generate(strpos)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
