"""Download NaijaVoices corpus.

NaijaVoices requires manual registration at https://naijavoices.org/.
This script prints instructions and the expected directory layout.
"""

print("""
NaijaVoices download instructions
==================================
1. Register at https://naijavoices.org/ and request download access.
2. Download the corpus zip/tar and extract to:

   workspace/data/raw/naijavoices/

   Expected layout:
     naijavoices/
       en/
         speaker_*/
           *.wav

3. Re-run: make build-manifests
""")
