# Spoken Command Dataset

- **100 speakers:** 84 foreign and 16 Filipino-English.
- **13 fixed intents:** 3 phrases each.
- **6 variable intents:** 3 phrase templates x 3 slot values = 9 utterances each.
- **Two acoustic conditions:** clean and light background noise.
- **18,600 original files; 17,851 remaining after filtering.**
- **Transcriber used:** [simple-audio-transcriber by Martinnavs](https://github.com/Martinnavs/simple-audio-transcriber).

## Dataset overview

| Item | Count |
|---|---:|
| Reference speakers | 100 |
| Foreign speakers ([LibriSpeech](https://www.openslr.org/12), [small subsets](https://www.openslr.org/31)) | 84 |
| Filipino-English speakers ([SilencioPH](https://huggingface.co/datasets/SilencioNetwork/tagalog-filipino-speech)) | 16 |
| Intents | 19 |
| Intents without slots | 13 |
| Intents with slots | 6 |
| Acoustic conditions per utterance | 2 |
| Active clean WAV files | 8,927 |
| Active noisy WAV files | 8,924 |
| Original WAV files | 18,600 |
| Excluded WAV paths | 749 |
| WAV files stored in FLAGGED | 749 |
| **Remaining active WAV files** | **17,851** |

*DGX2 counts as of 2026-10-01, for the command dataset described here.*

## Speakers

- All 100 speaker IDs remain after filtering.
- The table shows original file counts.

| Group | Speaker IDs | Speakers | WAV files |
|---|---|---:|---:|
| Foreign (LibriSpeech) | `s1-s67`, `s81-s88`, `s91-s99` | 84 | 15,624 |
| Filipino-English (SilencioPH) | `s68-s80`, `s89-s90`, `s100` | 16 | 2,976 |
| **Total** | **`s1-s100`** | **100** | **18,600** |

| Metadata split | Foreign speaker IDs | Filipino speaker IDs | Total speakers |
|---|---|---|---:|
| Train | `s1-s67` | `s68-s80` | 80 |
| Validation | `s81-s88` | `s89-s90` | 10 |
| Test | `s91-s99` | `s100` | 10 |

## Acoustic conditions

| Filename suffix | Condition | Files |
|---|---|---:|
| `_clean.wav` | Clean speech | 8,927 |
| `_noisy.wav` | Speech with light background noise | 8,924 |

## Phrase variations

| Group | Intent | v1 | v2 | v3 |
|---|---|---|---|---|
| Music control | `PLAY_MUSIC` | Play music | Start music | Play some music |
| Music control | `VOLUME_UP` | Volume up | Increase the volume | Turn the volume up |
| Music control | `VOLUME_DOWN` | Volume down | Lower the volume | Turn the volume down |
| Music control | `NEXT` | Next song | Skip song | Play next song |
| Music control | `PAUSE` | Pause | Pause audio | Pause song |
| Music control | `STOP` | Stop | Stop playing | End playback |
| Lighting | `LIGHT_ON` | Lights on | Power on the lights | Turn on the lights |
| Lighting | `LIGHT_OFF` | Lights out | Kill the lights | Shut off the lights |
| Lighting | `BRIGHTNESS` | Brightness {percent} | Adjust brightness to {percent} | Brightness level {percent} |
| Lighting | `COLOR` | Change color to {color} | Switch color to {color} | Set color to {color} |
| Temperature control | `TEMPERATURE` | Temperature {degrees} | Change the temperature to {degrees} | Set the temperature to {degrees} |
| Information | `WEATHER` | Weather | What's the weather? | Tell me the weather |
| Information | `TIME` | Time | What time is it? | Tell me the time |
| Timers and alarms | `TIMER` | Timer {duration} | Countdown for {duration} | Start a timer for {duration} |
| Timers and alarms | `ALARM` | Alarm {time} | Wake me up at {time} | Set an alarm for {time} |
| Communication | `CALL` | Call | Make a call | Make a phone call |
| Communication | `MESSAGE` | Message | Send a message | Send my message |
| Reminders | `CREATE_REMINDER` | Reminder {task} | Remind me to {task} | Create a reminder to {task} |
| Reminders | `LIST_REMINDERS` | Reminders | Show my reminders | List my reminders |

### Slot values

| Intent            | Slot         | Values                              |
| ----------------- | ------------ | ----------------------------------- |
| `TIMER`           | `{duration}` | 10 seconds; 30 seconds; 1 minute    |
| `ALARM`           | `{time}`     | 6 AM; 8 AM; 9 PM                    |
| `TEMPERATURE`     | `{degrees}`  | 18 degrees; 22 degrees; 26 degrees  |
| `BRIGHTNESS`      | `{percent}`  | 20 percent; 60 percent; 100 percent |
| `COLOR`           | `{color}`    | red; blue; green            |
| `CREATE_REMINDER` | `{task}`     | drink water; study; exercise       |

## Transcription-Based Filtering Results

Files are flagged individually when transcription similarity is **below 0.80**.

| Intent / slot folder | Original WAV files | Excluded WAV paths | Remaining active WAV files |
|---|---:|---:|---:|
| `ALARM_6_00AM` | 600 | 17 | 583 |
| `ALARM_8_00AM` | 600 | 17 | 583 |
| `ALARM_9_00PM` | 600 | 19 | 581 |
| `BRIGHTNESS_100` | 600 | 3 | 597 |
| `BRIGHTNESS_20` | 600 | 5 | 595 |
| `BRIGHTNESS_60` | 600 | 5 | 595 |
| `CALL` | 600 | 63 | 537 |
| `COLOR_BLUE` | 600 | 2 | 598 |
| `COLOR_GREEN` | 600 | 4 | 596 |
| `COLOR_RED` | 600 | 5 | 595 |
| `CREATE_REMINDER_DRINK_WATER` | 600 | 4 | 596 |
| `CREATE_REMINDER_EXERCISE` | 600 | 10 | 590 |
| `CREATE_REMINDER_STUDY` | 600 | 25 | 575 |
| `LIGHT_OFF` | 600 | 49 | 551 |
| `LIGHT_ON` | 600 | 34 | 566 |
| `LIST_REMINDERS` | 600 | 48 | 552 |
| `MESSAGE` | 600 | 75 | 525 |
| `NEXT` | 600 | 11 | 589 |
| `PAUSE` | 600 | 46 | 554 |
| `PLAY_MUSIC` | 600 | 10 | 590 |
| `STOP` | 600 | 52 | 548 |
| `TEMPERATURE_18` | 600 | 2 | 598 |
| `TEMPERATURE_22` | 600 | 1 | 599 |
| `TEMPERATURE_26` | 600 | 2 | 598 |
| `TIME` | 600 | 56 | 544 |
| `TIMER_10s` | 600 | 13 | 587 |
| `TIMER_1m` | 600 | 38 | 562 |
| `TIMER_30s` | 600 | 8 | 592 |
| `VOLUME_DOWN` | 600 | 25 | 575 |
| `VOLUME_UP` | 600 | 39 | 561 |
| `WEATHER` | 600 | 61 | 539 |
| **Total** | **18,600** | **749** | **17,851** |

## Filename convention

`<FOLDER_NAME>_s<speaker>_v<phrase variation>_<condition>.wav`

Example: `BRIGHTNESS_100/BRIGHTNESS_100_s1_v3_noisy.wav`

| Component | Meaning |
|---|---|
| `BRIGHTNESS_100/` | Intent folder with the 100-percent slot value |
| `s1` | Reference speaker 1 |
| `v3` | Third phrase variation |
| `noisy` | Acoustic condition |

## Metadata

- `manifest.csv`: audio paths, intent, speaker, split, phrase, condition, transcript, slot value, and duration.
- `labels.json` and `slots.json`: intent and slot definitions.
- [FLAGGED/summary.md](FLAGGED/summary.md): excluded recordings and QA summary.
