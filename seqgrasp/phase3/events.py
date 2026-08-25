from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class HandoffEvent(StrEnum):
    ACQUISITION_CONTACT_ESTABLISHED = "ACQUISITION_CONTACT_ESTABLISHED"
    ADDITIONAL_FINGER_RECRUITED = "ADDITIONAL_FINGER_RECRUITED"
    PALM_CONTACT_ESTABLISHED = "PALM_CONTACT_ESTABLISHED"
    SUPPORT_LOAD_SHIFTED = "SUPPORT_LOAD_SHIFTED"
    ACQUISITION_FINGER_UNLOADED = "ACQUISITION_FINGER_UNLOADED"
    ACQUISITION_FINGER_RELEASED = "ACQUISITION_FINGER_RELEASED"
    RESOURCE_RECOVERED = "RESOURCE_RECOVERED"
    COMPLETE_OBJECT_LOSS = "COMPLETE_OBJECT_LOSS"


@dataclass
class HandoffEventDetector:
    previous_thumb_index_contact: bool = False
    previous_middle_contact: bool = False
    previous_palm_contact: bool = False

    def update(
        self,
        *,
        thumb_index_contact: bool,
        middle_contact: bool,
        palm_contact: bool,
        load_shifted: bool = False,
        acquisition_unloaded: bool = False,
        acquisition_released: bool = False,
        object_retained: bool = False,
        alternate_support: bool = False,
        released_finger_has_motion: bool = False,
        complete_object_loss: bool = False,
    ) -> tuple[HandoffEvent, ...]:
        events: list[HandoffEvent] = []
        if thumb_index_contact and not self.previous_thumb_index_contact:
            events.append(HandoffEvent.ACQUISITION_CONTACT_ESTABLISHED)
        if middle_contact and not self.previous_middle_contact:
            events.append(HandoffEvent.ADDITIONAL_FINGER_RECRUITED)
        if palm_contact and not self.previous_palm_contact:
            events.append(HandoffEvent.PALM_CONTACT_ESTABLISHED)
        if load_shifted:
            events.append(HandoffEvent.SUPPORT_LOAD_SHIFTED)
        if acquisition_unloaded:
            events.append(HandoffEvent.ACQUISITION_FINGER_UNLOADED)
        if acquisition_released:
            events.append(HandoffEvent.ACQUISITION_FINGER_RELEASED)
        if acquisition_released and object_retained and alternate_support and released_finger_has_motion:
            events.append(HandoffEvent.RESOURCE_RECOVERED)
        if complete_object_loss:
            events.append(HandoffEvent.COMPLETE_OBJECT_LOSS)
        self.previous_thumb_index_contact = thumb_index_contact
        self.previous_middle_contact = middle_contact
        self.previous_palm_contact = palm_contact
        return tuple(events)
