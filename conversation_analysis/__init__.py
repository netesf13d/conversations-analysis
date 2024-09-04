# -*- coding: utf-8 -*-
"""
TODO doc
This package implements functionality to load, modify and analyze conversations


It features:
* Conversation
    - Facebook Messenger
    - Whatsapp
* Conversation analysis
* Data plotting
"""

from .conversations import (MessengerConversation,
                            list_messenger_conversations,
                            WhatsappConversation)
from .analysis import ConversationStats, word_count_dataframe
from .plot import pie_plot, bar_plot, stack_plot

