import { getGuildState } from "./guildState.svelte";
import { writable, get } from "svelte/store";

// Create writable stores for state management
const semanticDistances = writable(new Map<string, number>());
const currentGuildId = writable("");
const currentChannelId = writable("");

export function getSemanticBoundaryState() {
    const guildState = getGuildState();
    
    // Track changes to guild/channel
    function checkChannelChange() {
        const guildStateId = guildState.guildId;
        const channelStateId = guildState.channelId;
        const currGuildId = get(currentGuildId);
        const currChannelId = get(currentChannelId);
        
        if (channelStateId !== currChannelId || guildStateId !== currGuildId) {
            // Channel changed, clear semantic distances
            semanticDistances.set(new Map());
            currentChannelId.set(channelStateId || "");
            currentGuildId.set(guildStateId);
        }
    }
    
    // Initial check
    checkChannelChange();

    /**
     * Set semantic distances for a batch of messages
     * @param distances Array of {messageId, nextMessageId, distance} objects
     */
    function setDistances(distances: Array<{messageId: string, nextMessageId: string, distance: number}>) {
        console.log("Setting semantic distances:", distances.length, "items");
        semanticDistances.update(map => {
            const newMap = new Map(map);
            for (const item of distances) {
                newMap.set(item.messageId, item.distance);
            }
            return newMap;
        });
    }

    /**
     * Get boundary strength as a normalized value (0-1)
     * Used for visualization intensity
     * @param messageId ID of the message
     */
    function getBoundaryStrength(messageId: string): number {
        const distancesMap = get(semanticDistances);
        
        if (!distancesMap.has(messageId)) {
            return 0;
        }
        
        const distance = distancesMap.get(messageId) || 0;
        const values = Array.from(distancesMap.values());
        const maxDistance = values.length > 0 ? Math.max(...values) : 0;
        
        // Normalize distance to 0-1 range
        return maxDistance > 0 ? distance / maxDistance : 0;
    }
    
    /**
     * Get raw semantic distance value for a message
     * @param messageId ID of the message
     */
    function getDistance(messageId: string): number {
        const distancesMap = get(semanticDistances);
        return distancesMap.get(messageId) || 0;
    }

    return {
        // Methods
        setDistances,
        getBoundaryStrength,
        getDistance,
        checkChannelChange,
        
        // Legacy properties needed by other components
        get currentChannelId() { return get(currentChannelId); },
        get currentGuildId() { return get(currentGuildId); }
    };
}
