# VectorSearch-VideoText
Learn how to search video within text using SDDB and Vector embeddings

lets go!!
```
!pip install superduperdb
!pip install opencv-python
!pip install git+https://github.com/openai/CLIP.git
```

```
import clip
from IPython.display import *
from PIL import Image
import torch

from superduperdb import CFG
from superduperdb.ext.pillow import pil_image
from superduperdb.base.document import Document as D
from superduperdb import Model, Schema
from superduperdb.backends.mongodb.query import Collection
from superduperdb.ext.torch import tensor, TorchModel
```

lets Make the database superduper!

```
import os

# Uncomment one of the following lines to use a bespoke MongoDB deployment
# For testing the default connection is to mongomock

mongodb_uri = os.getenv("MONGODB_URI","mongomock://test")
# mongodb_uri = "mongodb://localhost:27017"
# mongodb_uri = "mongodb://superduper:superduper@mongodb:27017/documents"
# mongodb_uri = "mongodb://<user>:<pass>@<mongo_cluster>/<database>"
# mongodb_uri = "mongodb+srv://<username>:<password>@<atlas_cluster>/<database>"

CFG.downloads.hybrid = True
CFG.downloads.root = './'

# Super-Duper your Database!
from superduperdb import superduper
db = superduper(mongodb_uri)
```

```
from superduperdb import Encoder

vid_enc = Encoder(
    identifier='video_on_file',
    load_hybrid=False,
)

db.add(vid_enc)
```

Let's get a sample video from the net

```
db.execute(
    Collection('videos')
        .insert_one(
            D({'video': vid_enc(uri='https://superduperdb-public.s3.eu-west-1.amazonaws.com/animals_excerpt.mp4')})
        )
)
```

```
list(db.execute(Collection('videos').find()))
)
```

```
import cv2
import tqdm


def video2images(video_file):
    sample_freq = 10
    cap = cv2.VideoCapture(video_file)

    frame_count = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    extracted_frames = []
    progress = tqdm.tqdm()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_timestamp = frame_count // fps
        
        if frame_count % sample_freq == 0:
            extracted_frames.append({
                'image': Image.fromarray(frame[:,:,::-1]),
                'current_timestamp': current_timestamp,
            })
        frame_count += 1        
        progress.update(1)
    
    cap.release()
    cv2.destroyAllWindows()
    return extracted_frames
```

Create a Listener which will continously download video urls and save best frames into other collection.

from superduperdb import Listener

```
video2images = Model(
    identifier='video2images',
    object=video2images,
    flatten=True,
    model_update_kwargs={'document_embedded': False},
    output_schema=Schema(identifier='myschema', fields={'image': pil_image})
)

db.add(
   Listener(
       model=video2images,
       select=Collection('videos').find(),
       key='video',
   )
)
```

```
db.execute(Collection('_outputs.video.video2images').find_one()).unpack()['_outputs']['video']['video2images']['image']
```

```
model, preprocess = clip.load("RN50", device='cpu')
t = tensor(torch.float, shape=(1024,))

visual_model = TorchModel(
    identifier='clip_image',
    preprocess=preprocess,
    object=model.visual,
    encoder=t,
)
text_model = TorchModel(
    identifier='clip_text',
    object=model,
    preprocess=lambda x: clip.tokenize(x)[0],
    forward_method='encode_text',
    encoder=t,
    device='cpu',
    preferred_devices=None
)
```

Create VectorIndex with an indexing and compatible listener

```
from superduperdb import Listener, VectorIndex
from superduperdb.backends.mongodb import Collection

db.add(
    VectorIndex(
        identifier='video_search_index',
        indexing_listener=Listener(
            model=visual_model,
            key='_outputs.video.video2images.image',
            select=Collection('_outputs.video.video2images').find(),
        ),
        compatible_listener=Listener(
            model=text_model,
            key='text',
            select=None,
            active=False
        )
    )
)
```

Now lets Test vector search by quering a text against saved frames.

Search for something that may have happened during the video:

```
search_phrase = 'An elephant'

r = next(db.execute(
    Collection('_outputs.video.video2images').like(D({'text': 'An elephant'}), vector_index='video_search_index', n=1).find()
))

search_timestamp = r['_outputs']['video']['video2images']['current_timestamp']
```

Get the back-reference to the original video document:

```
video = db.execute(Collection('videos').find_one({'_id': r['_source']}))
```

Start the video from the resultant timestamp:

```
from IPython.display import display, HTML
video_html = f"""
<video width="640" height="480" controls>
    <source src="{video['video'].uri}" type="video/mp4">
</video>
<script>
    var video = document.querySelector('video');
    video.currentTime = {search_timestamp};
    video.play();
</script>
"""

display(HTML(video_html))
```










