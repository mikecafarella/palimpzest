from palimpzest.elements import DataRecord, UserFunction
from palimpzest.datamanager import DataDirectory
from palimpzest.corelib import URL, WebPage, Download

import requests
import datetime
from bs4 import BeautifulSoup

#
# REMIND: I don't love the syntax of UDFs right now. 
# It's too verbose and clumsy.
#


class DownloadHTMLFunction(UserFunction):
    """DownloadHTMLFunction downloads the HTML content of a web page."""
    def __init__(self):
        super().__init__("localfunction:webgetter", URL, WebPage)

    def html_to_text_with_links(self, html):
        # Parse the HTML content
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all hyperlink tags
        for a in soup.find_all('a'):
            # Check if the hyperlink tag has an 'href' attribute
            if a.has_attr('href'):
                # Replace the hyperlink with its text and URL in parentheses
                a.replace_with(f"{a.text} ({a['href']})")
        
        # Extract text from the modified HTML
        text = soup.get_text(separator='\n', strip=True)        
        return text

    # Someday we should introduce an abstraction that lets us
    # coalesce many requests into a bulk operation. This code is
    # fine for a dozen requests, but not for a million.
    def map(self, dr: DataRecord):
        textcontent = requests.get(dr.url).text
        dr2 = DataRecord(self.outputSchema, parent_uuid=dr._uuid)
        dr2.url = dr.url

        html = textcontent
        tokens = html.split()[:5000]
        dr2.html = " ".join(tokens)

        strippedHtml = self.html_to_text_with_links(textcontent)
        tokens = strippedHtml.split()[:5000]
        dr2.text = " ".join(tokens)

        # get current timestamp, in nice ISO format
        dr2.timestamp = datetime.datetime.now().isoformat()
        return dr2
    
class DownloadBinaryFunction(UserFunction):
    """DownloadBinaryFunction downloads binary content from a URL."""
    def __init__(self):
        super().__init__("localfunction:downloader", URL, Download)

    # Someday we should introduce an abstraction that lets us
    # coalesce many requests into a bulk operation. This code is
    # fine for a dozen requests, but not for a million.
    def map(self, dr: DataRecord):
        content = requests.get(dr.url).content
        dr2 = DataRecord(self.outputSchema, parent_uuid=dr._uuid)
        dr2.url = dr.url
        dr2.content = content
        dr2.timestamp = datetime.datetime.now().isoformat()
        return dr2

DataDirectory().registerUserFunction(DownloadHTMLFunction())
DataDirectory().registerUserFunction(DownloadBinaryFunction())

