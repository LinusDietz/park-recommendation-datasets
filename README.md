# Park Recommendation Datasets


Column Descriptions:

- `user` - hashed user ID for anonymization
- `park` - park id from OpenStreetMap, `https://www.openstreetmap.org/way/<ID>` or `https://www.openstreetmap.org/relation/<ID>`
- `rating` - always 1 (implicit data)
- `timestamp` - unix timestamp, not available in the survey dataset
- `activity` - the activity the user was involved in

## Flickr

- `flickr.csv` - visits of parks for activities
- `flickr_users.csv` - locations of users (computed as center of geographic interest)

 ## Survey

- `survey.csv` - mentions of parks for activities in the survey 
- `survey_mentions.csv` - locations of users (by self-reported postal area)
