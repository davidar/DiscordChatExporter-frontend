<script lang="ts">
    import { isDateDifferent } from "../js/helpers";
    import { fetchMessages, fetchSemanticDistances } from "../js/stores/api";
    import { getGuildState } from "../js/stores/guildState.svelte";
    import { getLayoutState } from "../js/stores/layoutState.svelte";
    import { getSemanticBoundaryState } from "../js/stores/semanticBoundaryStore";
    import DateSeparator from "./DateSeparator.svelte";
    import InfiniteScroll3 from "./InfiniteScroll3.svelte";
    import ChannelStart from "./message/ChannelStart.svelte";
    import Message from "./message/Message.svelte";
    import SemanticBoundarySeparator from "./SemanticBoundarySeparator.svelte";
    import type { Message as MessageType } from "../js/interfaces";

    const guildState = getGuildState()
    const layoutState = getLayoutState()
    const semanticBoundaryState = getSemanticBoundaryState();

    let apiGuildId = $derived(guildState.guildId ? guildState.guildId : "000000000000000000000000")
    let apiChannelId = $derived(guildState.channelId)

    async function fetchMessagesWrapper(direction: "before" | "after" | "around" | "first" | "last", messageId: string | null = null, limit: number) {
        const messagesResponse = await fetchMessages(apiGuildId, apiChannelId || "", direction, messageId, limit);
        
        // Always fetch semantic distances
        if (apiChannelId) {
            console.log("Fetching semantic distances for", direction, messageId);
            try {
                const distancesResponse = await fetchSemanticDistances(apiGuildId, apiChannelId, direction, messageId, limit);
                
                if (distancesResponse && distancesResponse.messageDistances) {
                    console.log("Received semantic distances:", distancesResponse.messageDistances.length, "items");
                    semanticBoundaryState.setDistances(distancesResponse.messageDistances);
                } else {
                    console.error("No messageDistances found in response:", distancesResponse);
                }
            } catch (error) {
                console.error("Error fetching semantic distances:", error);
            }
        }
        
        return messagesResponse;
    }
</script>

{#snippet channelStartSnippet(message)}
    <ChannelStart channelName={message.channelName} isThread={false} messageAuthor={message.author} />
{/snippet}

{#snippet channelEndSnippet(index, message, previousMessage)}
    <div data-messageid="last">
        this is the end of the channel
    </div>
{/snippet}

{#snippet renderMessageSnippet2(message, previousMessage)}
    <div data-messageid={message._id}>
        {#if message._id === "first"}
            <div>channel start</div>
        {:else if message._id === "last"}
            <div>channel end</div>
        {:else}
            {#if isDateDifferent(previousMessage, message)}
                <DateSeparator messageId={message._id} />
            {/if}
            
            <SemanticBoundarySeparator messageId={message._id} />
            
            <Message message={message} previousMessage={previousMessage} />
        {/if}
    </div>
{/snippet}

<div class="channel-wrapper" class:threadshown={layoutState.threadshown}>
    <div class="channel" >
        {#if apiChannelId}
            <!-- TODO: support change of selectedMessageId without rerender -->
            {#key guildState.channelMessageId}
                {#key apiChannelId}
                    <InfiniteScroll3
                        fetchMessages={fetchMessagesWrapper}
                        scrollToMessageId={guildState.channelMessageId || ""}
                        snippetMessage={renderMessageSnippet2}
                        channelStartSnippet={channelStartSnippet}
                    />
                {/key}
            {/key}
        {/if}
    </div>
</div>


<style>
    .channel-wrapper {
        height: 100%;
        overflow: hidden;
    }

    .threadshown {
        border-top-right-radius: 8px;
    }
    .channel {
        background-color: #313338;
        height: 100%;
    }
</style>